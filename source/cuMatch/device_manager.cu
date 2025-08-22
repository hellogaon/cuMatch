#include <cu_match/device_manager.cuh>
#include <cu_match/device_defines.cuh>
#include <cu_match/cache_controller.h>
#include <gstream/cuda_api.h>
#include <gstream/grid_format/extent.h>
#include <gstream/grid_format/flip24_shard.h>
#include <gstream/grid_format/dense_block.h>
#include <omp.h>
#include <ash/stop_watch.h>
#include <cassert>
#include <thread>

namespace cu_match {

device_manager::device_manager() {}

device_manager::~device_manager() {
    _cleanup();
}

void device_manager::init(config_t const& cfg) {
    _cfg = cfg;
    _num_query_vertices = _cfg.qgraph->num_query_vertices();
    _num_query_edges = _cfg.qgraph->num_query_edges();
    _num_constraints = _cfg.qgraph->num_constraints();
    _num_max_negative_vertices = _cfg.qgraph->num_max_negative_vertices();
    cudaGetDeviceCount(&_num_gpus);
    assert(_num_gpus <= MaxNumGPUs);

    _vertices_info = _cfg.qgraph->qgraph_vertices_info();
    _edges_info = _cfg.qgraph->qgraph_edges_info();
    _gk_info = new global_kernel_task_info;
    _aligned_hostbuf_size = ash::aligned_size(_cfg.host_buffer_size, DiskIOSectorSize);
    _aligned_devbuf_size = ash::aligned_size(_cfg.device_buffer_size, DiskIOSectorSize * 2); // for double buffering
    _max_header_size = _get_max_header_size();
    assert(_max_header_size < _aligned_hostbuf_size);

    {
        void* pinned_cache_buffer = gstream::cuda::pinned_malloc(_aligned_hostbuf_size);
    
        _mmgmt.host_buddy = new ash::portable_buddy_system();
        _mmgmt.host_buddy->init(ash::memory_region{ pinned_cache_buffer, _aligned_hostbuf_size }, DiskIOSectorSize);

        _mmgmt.malloc_header_buffer = [this](uint64_t size) {
            return _mmgmt.host_buddy->allocate(size);
        };

        _mmgmt.malloc_shard_buffer = [this](sleaf_ptr const& sleaf) {
            return _mmgmt.host_buddy->allocate(sleaf->size_info.in_memory_size);
        };

        _mmgmt.free_buffer = [this](void* p) {
            return _mmgmt.host_buddy->deallocate(p);
        };

        _mmgmt.acquire_memory_space = [this]() {
            return _evict_cached_shard_once();
        };

        for (uint32_t g_id = 0; g_id < _num_gpus; g_id++) {
            _host_header_buffer[g_id] = _mmgmt.malloc_header_buffer(_max_header_size);
        }
    }
    {
        shard_cachectl::config_t cctl_cfg;
        cctl_cfg.num_entities = _cfg.gfs->num_shards();

        _mmgmt.shard_cctl = new shard_cachectl();
        if (!_mmgmt.shard_cctl->init(cctl_cfg)) {
            printf("mmgmt initialize error\n");
            exit(1);
        }
    }

    {
        if (_cfg.in_memory_mode) {
            using namespace gstream;
            using namespace grid_format;
            grid_file_stream const& gfs = *_cfg.gfs;

            for (uint32_t i = 0; i < _num_query_edges; i++) {
                uint32_t v1_label_id = _vertices_info[_edges_info[i].v1_id].vl_id;
                uint32_t v2_label_id = _vertices_info[_edges_info[i].v2_id].vl_id;
                uint32_t el_id = _edges_info[i].el_id;

                grid_off v1_min_gid = _cfg.gfs->get_min_grid_row(v1_label_id);
                grid_off v1_max_gid = _cfg.gfs->get_max_grid_row(v1_label_id);
                grid_off v2_min_gid = _cfg.gfs->get_min_grid_row(v2_label_id);
                grid_off v2_max_gid = _cfg.gfs->get_max_grid_row(v2_label_id);
                
                for (uint32_t r = v1_min_gid; r < v1_max_gid; r++) {
                    for (uint32_t c = v2_min_gid; c < v2_max_gid; c++) {
                        gbid_t gbids[2] = { make_gbid(r, c, el_id), make_gbid(c, r, el_id) };

                        for (uint32_t j = 0; j < 2; j++) {
                            gbid_t gbid = gbids[j];
                            stree_cptr const stree = gfs.stree(gbid);

                            if (stree == nullptr)
                                continue;

                            stree->post_order_dfs_without_leaf([&](stree_node_ptr xnode) {
                                if (xnode->header.qnode_type == qtree_node_type::InternalNode)
                                    return;
                                
                                stree_switch_node& sswch = *xnode;
                                
                                sleaf_ptr sleaf_arr[2] = { sswch.out_edges, sswch.in_edges };

                                for (uint32_t k = 0; k < 2; k++) {
                                    sleaf_ptr& sleaf = sleaf_arr[k];
                                    uint64_t const shard_size = sleaf->size_info.in_memory_size;
                                    
                                    if (!_mmgmt.shard_cctl->test(sleaf->unique_id)) {
                                        void* p = _mmgmt.malloc_shard_buffer(sleaf);
                                        assert(p != nullptr && "host buffer size is small");

                                        bool load_rslt = gfs.load_in_memory_shard(p, sleaf);
                                        assert(load_rslt);

                                        _mmgmt.shard_cctl->add(sleaf->unique_id, { p, shard_size } );
                                    }
                                }
                            });

                            if (_edges_info[j].e_dir != qgraph_edge_direction::UNDIRECTED) // skip
                                break;
                        }
                    }
                }  
            }

        }
    }
    {
        for (uint32_t g_id = 0; g_id < _num_gpus; g_id++) {
            cudaSetDevice(g_id);
            _device_buffer[g_id] = nullptr;
            _device_buffer[g_id] = gstream::cuda::device_malloc(_aligned_devbuf_size);
            gstream::cuda::device_memset(_device_buffer[g_id], 0, _aligned_devbuf_size);
        }
    }

    {
        kernel_executor::config_t kexec_cfg;
        kexec_cfg.q_type = NORMAL_PATTERN;
        if (_num_constraints > 0) kexec_cfg.q_type |= CONSTRAINT_PATTERN;
        if (_cfg.qgraph->is_optional_pattern()) kexec_cfg.q_type |= OPTIONAL_PATTERN;
        if (_cfg.qgraph->is_negative_pattern()) kexec_cfg.q_type |= NEGATIVE_PATTERN;

        kexec_cfg.j_type = (!_cfg.is_table_join ? join_algorithm_type::ATTRIBUTE_JOIN : join_algorithm_type::TABLE_JOIN);

        _kexec = new kernel_executor;
        _kexec->init(kexec_cfg);
    }
}

GSTREAM_HOST_ONLY
bool device_manager::execute_match(lgf_scheduler* lgf_sched) {
    uint64_t total_counter = 0;
    uint64_t const max_left_buffer_size = _aligned_devbuf_size;
    sched_plan_tree* sptree = lgf_sched->get_sched_plan_tree();
    lgf_sched->init_global_kernel_info(_gk_info);
    
    ash::stop_watch watch1[MaxNumGPUs], watch2[MaxNumGPUs];
    double kernel_time[MaxNumGPUs] = {}, H2D_time[MaxNumGPUs] = {};

    bool first_task = true;
    while (true) {
        single_kernel_task_info* k_info = new single_kernel_task_info;
        uint64_t used_estimate = sptree->pop_single_kernel_task(k_info, max_left_buffer_size - _max_header_size, first_task);
        if (!used_estimate) break;
        _task_queue.push(k_info);
        _sleaf_vecs.push_back(&k_info->total_sleaf_ptrs);
        first_task = false;
    }

    if (_num_gpus > 1) {
        _make_shard_list();
        std::thread graph_loader(&device_manager::_graph_load_func, this);
        graph_loader.detach();
    }

#pragma omp parallel for
    for (uint32_t g_id = 0; g_id < _num_gpus; g_id++) {    
        kernel_args kargs;
        cudaStream_t stream;

        cudaSetDevice(g_id);
        kargs.buffer = _device_buffer[g_id];
        cudaStreamCreate(&stream);

        uint32_t iteration = 0;
        printf("> kernel task info:\n");
        printf("\tleft_buffer_size: %lu\n", max_left_buffer_size);
        printf("-------------------------------------------------------------------\n");

        while(true) {
            std::optional<single_kernel_task_info*> pop_task = pop_queue_front();
            if (!pop_task)
                break;

            single_kernel_task_info* k_info = *pop_task;
            uint64_t const kernel_header_size = _make_header_buffer(k_info, g_id, 0);
            assert(kernel_header_size < _max_header_size);
            watch2[g_id].reset();
            uint64_t const kernel_data_size = _make_data_buffer_with_H2D(k_info, kernel_header_size, stream, g_id, 0);
            assert(max_left_buffer_size > kernel_header_size + kernel_data_size);
            // _print_kernel_task_info(k_info);

            // H2D
            _kernel_H2D(kernel_header_size, stream, g_id, 0);
            gstream::cuda::stream_synchronize(stream);
            watch2[g_id].lab();
            
            H2D_time[g_id] += watch2[g_id].elapsed_sec();

            // Kernel
            printf("> GPU: %u, kernel: %u, kernel_buffer_size: %lu, num_slices: %lu\n", g_id, iteration, kernel_header_size + kernel_data_size, k_info->total_sleaf_ptrs.size());
            printf("-------------------------------------------------------------------\n");
            watch1[g_id].reset();

            uint64_t merged_subtask_counter = 0;
            uint32_t num_merged_subtask = k_info->tree_root_id.size();
            
            _kexec->LFTJ_kernel_invoker(kargs, stream);
            gstream::cuda::stream_synchronize(stream);
            _kernel_D2H(merged_subtask_counter, stream, g_id, 0);
            gstream::cuda::stream_synchronize(stream);
            printf("> merged_subtask_num: %-3u | counter: %lu/%lu\n", num_merged_subtask, merged_subtask_counter, total_counter + merged_subtask_counter);

#pragma omp atomic
            total_counter += merged_subtask_counter;

            printf("-------------------------------------------------------------------\n");
            watch1[g_id].lab();

            kernel_time[g_id] += watch1[g_id].elapsed_sec();
            printf("GPU %u kernel time: %lf\n", g_id, watch1[g_id].elapsed_sec());
            
            delete k_info;
            iteration++;
        }
    }

    printf("> Total counter: %lu\n", total_counter);
    double max_kernel_time = 0, max_H2D_time = 0;
    for (uint32_t i = 0; i < _num_gpus; i++) {
        printf("> GPU %u kernel time: %.5lf seconds\n", i, kernel_time[i]);
        printf("> GPU %u H2D time: %.5lf seconds\n", i, H2D_time[i]);
        max_kernel_time = std::max(max_kernel_time, kernel_time[i]);
        max_H2D_time = std::max(max_H2D_time, H2D_time[i]);
    }
    printf("> Kernel elapsed time: %.5lf seconds\n", max_kernel_time);
    printf("> H2D elapsed time: %.5lf seconds\n", max_H2D_time);

    return true;
}


GSTREAM_HOST_ONLY
bool device_manager::execute_match_with_double_buffering(lgf_scheduler* lgf_sched) {
    uint64_t total_counter = 0;
    uint64_t const max_left_buffer_size = _aligned_devbuf_size / 2;
    _output_offset = max_left_buffer_size;
    
    sched_plan_tree* sptree = lgf_sched->get_sched_plan_tree();
    lgf_sched->init_global_kernel_info(_gk_info);
    
    ash::stop_watch watch[MaxNumGPUs];
    double kernel_time[MaxNumGPUs] = {};

    bool first_task = true;
    while (true) {
        single_kernel_task_info* k_info = new single_kernel_task_info;
        uint64_t used_estimate = sptree->pop_single_kernel_task(k_info, max_left_buffer_size - _max_header_size, first_task);
        if (!used_estimate) break;
        _task_queue.push(k_info);
        _sleaf_vecs.push_back(&k_info->total_sleaf_ptrs);
        first_task = false;
    }

    if (_num_gpus > 1) {
        _make_shard_list();
        std::thread graph_loader(&device_manager::_graph_load_func, this);
        graph_loader.detach();
    }

#pragma omp parallel for
    for (uint32_t g_id = 0; g_id < _num_gpus; g_id++) {    
        kernel_args kargs;
        cudaStream_t stream, stream2;

        cudaSetDevice(g_id);
        kargs.buffer = _device_buffer[g_id];
        cudaStreamCreate(&stream);
        cudaStreamCreate(&stream2);

        // stream 0 init
        std::optional<single_kernel_task_info*> pop_task = pop_queue_front();
        if (!pop_task)
            continue;

        single_kernel_task_info* k_info1 = *pop_task;

        uint32_t iteration = 0;
        printf("> kernel task info:\n");
        printf("\tleft_buffer_size: %lu\n", max_left_buffer_size);
        printf("-------------------------------------------------------------------\n");

        uint64_t kernel_header_size = _make_header_buffer(k_info1, g_id, 0);
        assert(kernel_header_size < _max_header_size);
        uint64_t kernel_data_size = _make_data_buffer_with_H2D(k_info1, kernel_header_size, stream, g_id, 0);
        assert(max_left_buffer_size > kernel_header_size + kernel_data_size);
        
        // _print_kernel_task_info(k_info);

        // stream 0 H2D
        _kernel_H2D(kernel_header_size, stream, g_id, 0);
        gstream::cuda::stream_synchronize(stream);
        
        while (true) {
            // stream 0 kernel
            printf("> GPU: %u, kernel: %u, kernel_buffer_size: %lu, num_slices: %lu\n", g_id, iteration, kernel_header_size + kernel_data_size, k_info1->total_sleaf_ptrs.size());
            printf("-------------------------------------------------------------------\n");
            watch[g_id].reset();
            
            uint64_t merged_subtask_counter = 0;
            uint32_t num_merged_subtask = k_info1->tree_root_id.size();
            kargs.buffer = _device_buffer[g_id];
            
            _kexec->LFTJ_kernel_invoker(kargs, stream);
            
            // stream 1 init
            pop_task = pop_queue_front();
        
            single_kernel_task_info* k_info2 = *pop_task;

            if (pop_task) {
                kernel_header_size = _make_header_buffer(k_info2, g_id, 1);
                assert(kernel_header_size < _max_header_size);
                kernel_data_size = _make_data_buffer_with_H2D(k_info2, kernel_header_size, stream2, g_id, 1);
                assert(max_left_buffer_size > kernel_header_size + kernel_data_size);
                // _print_kernel_task_info(k_info);

                // stream 1 H2D
                _kernel_H2D(kernel_header_size, stream2, g_id, 1);
            }
            
            // stream 0 D2H
            _kernel_D2H(merged_subtask_counter, stream, g_id, 0);
            gstream::cuda::stream_synchronize(stream2);    
            gstream::cuda::stream_synchronize(stream);
            printf("> merged_subtask_num: %-3u | counter: %lu/%lu\n", num_merged_subtask, merged_subtask_counter, total_counter + merged_subtask_counter);
#pragma omp atomic
            total_counter += merged_subtask_counter;

            printf("-------------------------------------------------------------------\n");
            watch[g_id].lab();

            kernel_time[g_id] += watch[g_id].elapsed_sec();
            printf("GPU %u kernel time: %lf\n", g_id, watch[g_id].elapsed_sec());
            
            if (!pop_task)
                break;

            delete k_info1;
        
            iteration++;

            // stream 1 kernel
            printf("> GPU: %u, kernel: %u, kernel_buffer_size: %lu, num_slices: %lu\n", g_id, iteration, kernel_header_size + kernel_data_size, k_info2->total_sleaf_ptrs.size());
            printf("-------------------------------------------------------------------\n");
            watch[g_id].reset();
            
            merged_subtask_counter = 0;
            num_merged_subtask = k_info2->tree_root_id.size();
            kargs.buffer = static_cast<char*>(_device_buffer[g_id]) + _output_offset;

            _kexec->LFTJ_kernel_invoker(kargs, stream2);
            
            // stream 0 init
            pop_task = pop_queue_front();
            
            k_info1 = *pop_task;
            
            if (pop_task) {
                kernel_header_size = _make_header_buffer(k_info1, g_id, 0);
                assert(kernel_header_size < _max_header_size);
                kernel_data_size = _make_data_buffer_with_H2D(k_info1, kernel_header_size, stream,  g_id, 0);
                assert(max_left_buffer_size > kernel_header_size + kernel_data_size);
                // _print_kernel_task_info(k_info);

                // stream 0 H2D
                _kernel_H2D(kernel_header_size, stream,  g_id, 0);
            }
            
            // stream 1 D2H
            _kernel_D2H(merged_subtask_counter, stream2,  g_id, 1);
            gstream::cuda::stream_synchronize(stream);
            gstream::cuda::stream_synchronize(stream2);
            printf("> merged_subtask_num: %-3u | counter: %lu/%lu\n", num_merged_subtask, merged_subtask_counter, total_counter + merged_subtask_counter);
#pragma omp atomic
            total_counter += merged_subtask_counter;

            printf("-------------------------------------------------------------------\n");
            watch[g_id].lab();

            kernel_time[g_id] += watch[g_id].elapsed_sec();
            printf("GPU %u kernel time: %lf\n", g_id, watch[g_id].elapsed_sec());
            
            if (!pop_task)
                break;

            delete k_info2;

            iteration++;
        }
    }

    printf("> Total counter: %lu\n", total_counter);
    double max_kernel_time = 0;
    for (uint32_t i = 0; i < _num_gpus; i++) {
        printf("> GPU %u kernel time: %.5lf seconds\n", i, kernel_time[i]);
        max_kernel_time = std::max(max_kernel_time, kernel_time[i]);
    }
    printf("> Kernel elapsed time: %.5lf seconds\n", max_kernel_time);

    return true;
}

// forwards
GSTREAM_HOST_ONLY
uint64_t device_manager::_make_header_buffer(single_kernel_task_info* k_info, uint32_t const g_id, uint32_t const stream_id) {
    uint64_t offset = 0;
    kernel_buffer_header* host_header = static_cast<kernel_buffer_header*>(_host_header_buffer[g_id]);
    uint64_t const header_size = ash::aligned_size(sizeof(kernel_buffer_header), CudaAlignment);
    void* host_data = static_cast<char*>(_host_header_buffer[g_id]);
    void* dev_data = static_cast<char*>(_device_buffer[g_id]) + stream_id * _output_offset;
    offset += header_size;
    _shard_mapper[g_id].clear();
    
    host_header->num_query_vertices = _num_query_vertices;
    host_header->num_query_edges = _num_query_edges;
    host_header->num_constraints = _num_constraints;
    host_header->num_subtasks = k_info->subtasks.size();
    host_header->num_merged_subtasks = k_info->tree_root_id.size();
    host_header->num_sched_tree_nodes = k_info->num_tree_nodes;
    host_header->num_shards = k_info->total_sleaf_ptrs.size();
    host_header->is_table_join = _cfg.is_table_join;
    host_header->counter = 0;

    if (host_header->num_shards >= MaxNumShardPerIter) {
        printf("Constant MaxNumShardPerIter must be greater than %u\n", host_header->num_shards);
        exit(1);
    }

    // shard_ptr 
    {
        using namespace gstream;
        using namespace grid_format;

        uint64_t const shard_ptr_size = host_header->num_shards * sizeof(flip24_shard_ptr);
        host_header->shard_ptr = static_cast<flip24_shard_ptr*>(_seek_pointer(dev_data, offset));
        flip24_shard_ptr* host_shard_ptr = static_cast<flip24_shard_ptr*>(_seek_pointer(host_data, offset));
        offset += ash::aligned_size(shard_ptr_size, CudaAlignment);

        for (uint32_t i = 0; i < host_header->num_shards; i++) {
            sleaf_ptr& sleaf = k_info->total_sleaf_ptrs[i];
            _shard_mapper[g_id][sleaf->unique_id] = i;
        }
    }

    // global_order
    {
        uint64_t const global_order_size = _num_query_vertices * sizeof(uint32_t);
        host_header->global_order = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_global_order = _seek_pointer(host_data, offset);
        memcpy(host_global_order, _gk_info->global_order, global_order_size);
        offset += ash::aligned_size(global_order_size, CudaAlignment);
    }

    // prefix_sum_degree
    {
        uint64_t const prefix_sum_degree_size = (host_header->num_query_vertices + 1) * sizeof(uint32_t);
        host_header->prefix_sum_degree = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_prefix_sum_degree = _seek_pointer(host_data, offset);
        memcpy(host_prefix_sum_degree, _gk_info->prefix_sum_degree, prefix_sum_degree_size);
        offset += ash::aligned_size(prefix_sum_degree_size, CudaAlignment);
    }

    // eid_list
    {
        uint64_t const eid_list_size = _num_query_edges * 2 * sizeof(uint32_t);
        host_header->eid_list = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_eid_list = _seek_pointer(host_data, offset);
        memcpy(host_eid_list, _gk_info->eid_list, eid_list_size);
        offset += ash::aligned_size(eid_list_size, CudaAlignment);
    }

    // kernel_edge_type
    {
        uint64_t const edge_type_list_size = _num_query_edges * 2 * sizeof(kernel_edge_type);
        host_header->edge_type_list = static_cast<kernel_edge_type*>(_seek_pointer(dev_data, offset));
        void* host_edge_type_list = _seek_pointer(host_data, offset);
        memcpy(host_edge_type_list, _gk_info->edge_type_list, edge_type_list_size);
        offset += ash::aligned_size(edge_type_list_size, CudaAlignment);
    }

    // lv0_dep
    {
        uint64_t const lv0_dep_size = _num_query_edges * sizeof(uint32_t);
        host_header->lv0_dep = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_lv0_dep = _seek_pointer(host_data, offset);
        memcpy(host_lv0_dep, _gk_info->lv0_dep, lv0_dep_size);
        offset += ash::aligned_size(lv0_dep_size, CudaAlignment);
    }

    // constraints
    {
        uint64_t const constraint_ptr_size = (host_header->num_query_vertices + 1) * sizeof(uint32_t);
        host_header->constraint_ptr = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_constraint_ptr = _seek_pointer(host_data, offset);
        memcpy(host_constraint_ptr, _gk_info->constraint_ptr, constraint_ptr_size);
        offset += ash::aligned_size(constraint_ptr_size, CudaAlignment);

        uint64_t const constraints_size = host_header->num_constraints * sizeof(uint32_t);
        host_header->constraints = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_constraints = _seek_pointer(host_data, offset);
        memcpy(host_constraints, _gk_info->constraints, constraints_size);
        offset += ash::aligned_size(constraints_size, CudaAlignment);
    }

    // complex_info
    {
        uint64_t const num_complex_per_vertex_size = (host_header->num_query_vertices) * sizeof(uint32_t);
        host_header->num_optional_per_vertex = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_num_optional_per_vertex = _seek_pointer(host_data, offset);
        memcpy(host_num_optional_per_vertex, _gk_info->num_optional_per_vertex, num_complex_per_vertex_size);
        offset += ash::aligned_size(num_complex_per_vertex_size, CudaAlignment);

        host_header->num_negative_per_vertex = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_num_negative_per_vertex = _seek_pointer(host_data, offset);
        memcpy(host_num_negative_per_vertex, _gk_info->num_negative_per_vertex, num_complex_per_vertex_size);
        offset += ash::aligned_size(num_complex_per_vertex_size, CudaAlignment);
    }

    // is_optional_vertex
    {
        uint64_t const is_optional_vertex_size = _num_query_vertices * sizeof(bool);
        host_header->is_optional_vertex = static_cast<bool*>(_seek_pointer(dev_data, offset));
        void* host_is_optional_vertex = _seek_pointer(host_data, offset);
        memcpy(host_is_optional_vertex, _gk_info->is_optional_vertex, is_optional_vertex_size);
        offset += ash::aligned_size(is_optional_vertex_size, CudaAlignment);
    }

    // is_negative_vertex_with_degree
    {
        uint64_t const is_negative_vertex_with_degree_size = _num_query_vertices * sizeof(uint32_t);
        host_header->is_negative_vertex_with_degree = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_is_negative_vertex_with_degree = _seek_pointer(host_data, offset);
        memcpy(host_is_negative_vertex_with_degree, _gk_info->is_negative_vertex_with_degree, is_negative_vertex_with_degree_size);
        offset += ash::aligned_size(is_negative_vertex_with_degree_size, CudaAlignment);
    }

    // sched_tree
    {
        uint64_t const ptr_size = (host_header->num_sched_tree_nodes + 1) * sizeof(uint32_t);
        uint64_t const dst_size = (host_header->num_sched_tree_nodes - 1) * sizeof(uint32_t);

        host_header->sched_tree_ptr = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_sched_tree_ptr = _seek_pointer(host_data, offset);
        memcpy(host_sched_tree_ptr, k_info->sched_tree_ptr, ptr_size);
        offset += ash::aligned_size(ptr_size, CudaAlignment);

        host_header->sched_tree_nxt_id = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_sched_tree_nxt_id = _seek_pointer(host_data, offset);
        memcpy(host_sched_tree_nxt_id, k_info->sched_tree_nxt_id, dst_size);
        offset += ash::aligned_size(dst_size, CudaAlignment);

        host_header->sched_tree_prv_id = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_sched_tree_prv_id = _seek_pointer(host_data, offset);
        memcpy(host_sched_tree_prv_id, k_info->sched_tree_prv_id, dst_size);
        offset += ash::aligned_size(dst_size, CudaAlignment);

        host_header->sched_tree_st_id = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        void* host_sched_tree_st_id = _seek_pointer(host_data, offset);
        memcpy(host_sched_tree_st_id, k_info->sched_tree_st_id, dst_size);
        offset += ash::aligned_size(dst_size, CudaAlignment);
    }

    // tree_root_ids
    {
        uint64_t const tree_root_ids_size = (host_header->num_merged_subtasks) * sizeof(uint32_t);
        host_header->tree_root_ids = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
        uint32_t* host_tree_root_ids = static_cast<uint32_t*>(_seek_pointer(host_data, offset));
        for (uint32_t i = 0; i < host_header->num_merged_subtasks; i++)
            host_tree_root_ids[i] = k_info->tree_root_id[i];
        offset += ash::aligned_size(tree_root_ids_size, CudaAlignment);
    }
    
    // subtask_info
    {
        uint64_t const subtask_info_size = host_header->num_subtasks * sizeof(kernel_subtask_info);
        host_header->subtask_info = static_cast<kernel_subtask_info*>(_seek_pointer(dev_data, offset));
        kernel_subtask_info* host_sub_task_info = static_cast<kernel_subtask_info*>(_seek_pointer(host_data, offset));
        offset += ash::aligned_size(subtask_info_size, CudaAlignment); 
        
        uint64_t const shard_idx_list_size = _num_query_edges * 2 * sizeof(uint32_t);

        for (uint32_t i = 0; i < host_header->num_subtasks; i++) {
            host_sub_task_info[i].shard_idx_list = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
            uint32_t* host_shard_idx_list = static_cast<uint32_t*>(_seek_pointer(host_data, offset));
            offset += ash::aligned_size(shard_idx_list_size, CudaAlignment);

            for (uint32_t j = 0; j < _num_query_edges; j++) {
                uint64_t shard_id = k_info->subtasks[i]->sleaf_ptrs[j]->unique_id;
                uint32_t sleaf_idx = _shard_mapper[g_id][shard_id];
                if (!_cfg.is_table_join) {
                    host_shard_idx_list[_gk_info->v1_list_idxs[j]] = sleaf_idx;
                    host_shard_idx_list[_gk_info->v2_list_idxs[j]] = sleaf_idx;
                }
                else {
                    host_shard_idx_list[j] = sleaf_idx;
                }
            }
        }
    }

    // edge_order_with_vid (only for table join)
    {
        if (_cfg.is_table_join) {
            uint64_t const edge_order_with_vid_size = _num_query_edges * 2 * sizeof(uint32_t);
            host_header->edge_order_with_vid = static_cast<uint32_t*>(_seek_pointer(dev_data, offset));
            void* host_edge_order_with_vid = _seek_pointer(host_data, offset);
            memcpy(host_edge_order_with_vid, _gk_info->edge_order_with_vid, edge_order_with_vid_size);
            offset += ash::aligned_size(edge_order_with_vid_size, CudaAlignment);
        }
    }

    // is_join_vertex (only for table join)
    {
        if (_cfg.is_table_join) {
            uint64_t const is_join_vertex_size = _num_query_edges * 2 * sizeof(bool);
            host_header->is_join_vertex = static_cast<bool*>(_seek_pointer(dev_data, offset));
            void* host_is_join_vertex = _seek_pointer(host_data, offset);
            memcpy(host_is_join_vertex, _gk_info->is_join_vertex, is_join_vertex_size);
            offset += ash::aligned_size(is_join_vertex_size, CudaAlignment);
        }
    }

    host_header->header_size = offset;
    return offset;
}

uint64_t device_manager::_get_max_header_size() {
    uint64_t ret = 0;
    ret += ash::aligned_size(sizeof(kernel_buffer_header), CudaAlignment);
    ret += ash::aligned_size(MaxNumShardPerIter * sizeof(flip24_shard_ptr), CudaAlignment); // shard_ptr 
    ret += ash::aligned_size((MaxNumQueryVertices) * sizeof(uint32_t), CudaAlignment); // global_order
    ret += ash::aligned_size((MaxNumQueryVertices) * sizeof(bool), CudaAlignment); // is_optional_vertex
    ret += ash::aligned_size((MaxNumQueryVertices) * sizeof(uint32_t), CudaAlignment); // is_negative_vertex_with_degree
    ret += ash::aligned_size((MaxNumQueryVertices + 1) * sizeof(uint32_t), CudaAlignment); // prefix_sum_degree 
    ret += ash::aligned_size((MaxNumQueryEdges * 2) * sizeof(uint32_t), CudaAlignment); // eid_list
    ret += ash::aligned_size((MaxNumQueryEdges * 2) * sizeof(kernel_edge_type), CudaAlignment); // edge_type_list
    ret += ash::aligned_size((MaxNumQueryEdges) * sizeof(uint32_t), CudaAlignment); // lv0_dep
    ret += ash::aligned_size((MaxNumQueryVertices + 1) * sizeof(uint32_t), CudaAlignment); // constraint ptr
    ret += ash::aligned_size((MaxNumConstraints) * sizeof(uint32_t), CudaAlignment); // constraints
    ret += ash::aligned_size((MaxNumQueryVertices) * sizeof(uint32_t), CudaAlignment); // complex_info (optional)
    ret += ash::aligned_size((MaxNumQueryVertices) * sizeof(uint32_t), CudaAlignment); // complex_info (negative)
    ret += ash::aligned_size(MaxNumTreeNodes * sizeof(uint32_t), CudaAlignment);
    ret += ash::aligned_size(MaxNumTreeNodes * sizeof(uint32_t), CudaAlignment);
    ret += ash::aligned_size(MaxNumTreeNodes * sizeof(uint32_t), CudaAlignment);
    ret += ash::aligned_size(MaxNumTreeNodes * sizeof(uint32_t), CudaAlignment);
    ret += ash::aligned_size(MaxNumTreeNodes * sizeof(kernel_subtask_info), CudaAlignment);
    ret += ash::aligned_size(MaxNumTreeNodes * sizeof(uint32_t), CudaAlignment);
    for (uint32_t i = 0; i < MaxNumTreeNodes; i ++) {
        ret += ash::aligned_size((MaxNumQueryEdges * 2) * sizeof(uint32_t), CudaAlignment);
    }
    if (_cfg.is_table_join) {
        ret += ash::aligned_size((MaxNumQueryEdges * 2) * sizeof(uint32_t), CudaAlignment); // edge_order_with_vid
        ret += ash::aligned_size((MaxNumQueryEdges * 2) * sizeof(bool), CudaAlignment); // is_join_vertex
    }
    ret = ash::aligned_size(ret, gstream::DiskIoSectorSize);
    return ret;
}

uint64_t device_manager::_make_data_buffer_with_H2D(
            single_kernel_task_info* k_info, 
            uint64_t const start_offset, 
            gstream::cuda::stream_type const& stream, 
            uint32_t const g_id, 
            uint32_t const stream_id) {
            
    uint64_t offset = start_offset;
    kernel_buffer_header* host_header = static_cast<kernel_buffer_header*>(_host_header_buffer[g_id]);
    uint64_t const header_size = ash::aligned_size(sizeof(kernel_buffer_header), CudaAlignment);
    void* host_data = static_cast<char*>(_host_header_buffer[g_id]);
    void* dev_data = static_cast<char*>(_device_buffer[g_id]) + stream_id * _output_offset;

    flip24_shard_ptr* host_shard_ptr = static_cast<flip24_shard_ptr*>(_seek_pointer(host_data, header_size));

    {
        using namespace gstream;
        using namespace grid_format;

        offset = ash::aligned_size(offset, DiskIoSectorSize); // for direct I/O
        ash::stop_watch watch;
        for (uint32_t i = 0; i < host_header->num_shards; i++) {
            grid_file_stream const& gfs = *_cfg.gfs;
            sleaf_ptr& sleaf = k_info->total_sleaf_ptrs[i];
            uint64_t const shard_size = sleaf->size_info.in_memory_size;
            host_shard_ptr[i] = static_cast<flip24_shard_ptr>(_seek_pointer(dev_data, offset));

            if (_num_gpus == 1) {
                if (!_mmgmt.shard_cctl->test(sleaf->unique_id)) {
                    assert(!_cfg.in_memory_mode);
                    // load in cache
                    void* p = _allocate_shard_buffer_with_evcit(sleaf);
                    assert(p != nullptr && "host buffer size is small");

                    bool load_rslt = gfs.load_in_memory_shard(p, sleaf);
                    assert(load_rslt);
                    // printf("%u %u %u %lu %lu\n", shard->num_edges(), shard->gbid().row, shard->gbid().col, shard_size, sleaf->unique_id);
                    // flip24_shard_ptr shard = static_cast<flip24_shard_ptr>(host_shard);
                    _mmgmt.shard_cctl->add(sleaf->unique_id, { p, shard_size } );
                }
            }
            else {
                while (!_mmgmt.shard_cctl->test(sleaf->unique_id)) {
                    std::unique_lock<std::mutex> lock(_load_mutex);
                    _cv.wait(lock, [this] { return _ready.load(); });
                    _ready = false;
                }
            }
            
            shard_descriptor* shard_des = _mmgmt.shard_cctl->get(sleaf->unique_id);
            
            gstream::cuda::h2dcpy_async(
                reinterpret_cast<char*>(host_shard_ptr[i]),
                static_cast<char*>(shard_des->addr),
                shard_des->size,
                stream
            );
            
            offset += ash::aligned_size(shard_size, DiskIoSectorSize);
        }
        watch.lab();
        printf("GPU %u disk load time: %f\n", g_id, watch.elapsed_sec());
    }
    return offset;
}

void device_manager::_kernel_H2D(uint64_t size, gstream::cuda::stream_type const& stream,  uint32_t const g_id, uint32_t const stream_id) {
    gstream::cuda::h2dcpy_async(
        static_cast<char*>(_device_buffer[g_id]) + stream_id * _output_offset,
        static_cast<char*>(_host_header_buffer[g_id]),
        size,
        stream
    );
}

void device_manager::_kernel_D2H(uint64_t& output, gstream::cuda::stream_type const& stream,  uint32_t const g_id, uint32_t const stream_id) {
    kernel_buffer_header* dev = reinterpret_cast<kernel_buffer_header*>(static_cast<char*>(_device_buffer[g_id]) + stream_id * _output_offset);
    gstream::cuda::d2hcpy_async(
        &output,
        &(dev->counter),
        sizeof(uint64_t),
        stream
    );
}

void device_manager::_make_shard_list() {
    std::unordered_map<uint64_t, bool> _checker;
    for (std::vector<sleaf_ptr>* sleaf_ptrs: _sleaf_vecs) {
        uint64_t num_shards = sleaf_ptrs->size();
        for (uint32_t i = 0; i < num_shards; i++) {
            sleaf_ptr& sleaf = sleaf_ptrs->at(i);
            uint64_t u_id = sleaf->unique_id;
            if (_checker[u_id] == false) {
                _shard_list.push_back(sleaf);
                _checker[u_id] = true;
            }
        }
    }
}

void device_manager::_graph_load_func() {
    using namespace gstream;
    using namespace grid_format;
    grid_file_stream const& gfs = *_cfg.gfs;

    for (sleaf_ptr& sleaf: _shard_list) {
        uint64_t u_id = sleaf->unique_id;
        uint64_t const shard_size = sleaf->size_info.in_memory_size;
        if (!_mmgmt.shard_cctl->test(u_id)) {
            assert(!_cfg.in_memory_mode);

            void* p = _allocate_shard_buffer_with_evcit_for_multi(sleaf);
            assert(p != nullptr && "host buffer size is small");

            bool load_rslt = gfs.load_in_memory_shard(p, sleaf);
            assert(load_rslt);

            _mmgmt.shard_cctl->add(sleaf->unique_id, { p, shard_size } );
            std::unique_lock<std::mutex> lock(_load_mutex);
            _ready = true;
            _cv.notify_one();
        }
    }
}

void* device_manager::_allocate_shard_buffer_with_evcit(sleaf_ptr const& sleaf) const {
    void* p = nullptr;
    do {
        p = _mmgmt.malloc_shard_buffer(sleaf);
        if (p != nullptr)
            break;
        if (!_mmgmt.acquire_memory_space()) {
            break;
        }
    } while (true);
    return p;
}


void* device_manager::_allocate_shard_buffer_with_evcit_for_multi(sleaf_ptr const& sleaf) const {
    void* p = nullptr;
    do {
        p = _mmgmt.malloc_shard_buffer(sleaf);
        if (p != nullptr)
            break;
        printf("Not enough host memory size\n");
        printf("Sufficient host memory is required in multi-GPU environments (Just implementation issues)\n");
        exit(1);
        if (!_mmgmt.acquire_memory_space()) {
            break;
        }
    } while (true);
    return p;
}

bool device_manager::_evict_cached_shard_once() const {
    auto const [success, key, value] = _mmgmt.shard_cctl->evict();
    _ash_unused(key);
    if (!success)
        return false;
    assert(value.addr != nullptr);
    _free_shard_buffer(value.addr);
    return true;
}

void device_manager::_free_shard_buffer(void* p) const {
    _mmgmt.free_buffer(p);
}

void device_manager::_clear_cache() const {
    while (_evict_cached_shard_once());
}


void device_manager::_print_kernel_task_info(single_kernel_task_info* k_info) {
    uint32_t subtask_id = 0;
    printf("\t\tlocal_vertex_order: [ ");
    for (uint32_t i = 0; i < _num_query_vertices; i++)
        printf("%u ", _gk_info->global_order[i]);
    printf("]\n");
    printf("\t\tscheduling tree:\n");
    printf("\t\t\tsrc: [ ");
    for (uint32_t i = 0; i < k_info->num_tree_nodes; i++)
        printf("%u ", i);
    printf("]\n");
    printf("\t\t\tptr: [ ");
    for (uint32_t i = 0; i < k_info->num_tree_nodes + 1; i++)
        printf("%u ", k_info->sched_tree_ptr[i]);
    printf("]\n");
    printf("\t\t\tnxt_id: [ ");
    for (uint32_t i = 0; i < k_info->num_tree_nodes - 1; i++)
        printf("%u ", k_info->sched_tree_nxt_id[i]);
    printf("]\n");
    printf("\t\t\tprv_id: [ ");
    for (uint32_t i = 0; i < k_info->num_tree_nodes - 1; i++)
        printf("%u ", k_info->sched_tree_prv_id[i]);
    printf("]\n");
    printf("\t\t\tst_id: [ ");
    for (uint32_t i = 0; i < k_info->num_tree_nodes - 1; i++)
        printf("%u ", k_info->sched_tree_st_id[i]);
    printf("]\n");
    printf("\t\t\ttree_node_id: [ ");
    for (uint32_t i = 0; i < k_info->tree_root_id.size(); i++)
        printf("%u ", k_info->tree_root_id[i]);
    printf("]\n");

    for (subtask_info* st_info: k_info->subtasks) {
        printf("\t\tsubtask %u:\n", subtask_id++);
        printf("\t\t\tshard_id: [ ");
        for (uint32_t i = 0; i < _num_query_edges; i++)
            printf("%lu ", st_info->sleaf_ptrs[i]->unique_id);
        printf("]\n");
    }
    printf("\t\ttotal_shard: [ ");
    for (sleaf_ptr shard_ptr: k_info->total_sleaf_ptrs)
        printf("%lu ", shard_ptr->unique_id);
    printf("]\n");
}

void device_manager::_cleanup() {
    assert(_kexec != nullptr);
    delete _kexec;
    _kexec = nullptr;

    for (uint32_t g_id = 0; g_id < _num_gpus; g_id++) {
        _mmgmt.free_buffer(_host_header_buffer[g_id]);
        gstream::cuda::device_free(_device_buffer[g_id]);
    }

    _clear_cache();
    if (_mmgmt.host_buddy != nullptr) {
        if (!gstream::cuda::pinned_free(_mmgmt.host_buddy->rgn().ptr)) {
            printf("_mmgmt.host_buddy cuda::pinned_free() failure!");
        }
        delete _mmgmt.host_buddy;
        _mmgmt.host_buddy = nullptr;
    }
    delete _mmgmt.shard_cctl; _mmgmt.shard_cctl = nullptr;
    _mmgmt.malloc_shard_buffer = nullptr;
    _mmgmt.free_buffer = nullptr;
    delete _gk_info;
}

} // namespace cu_match
