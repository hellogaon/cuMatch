#ifndef CU_MATCH_DEVICE_MANAGER_CUH
#define CU_MATCH_DEVICE_MANAGER_CUH
#include <cu_match/device_defines.cuh>
#include <cu_match/kernel_executor.cuh>
#include <cu_match/subgraph_match_defines.h>
#include <cu_match/query_graph.h>
#include <cu_match/lgf_scheduler.h>
#include <cu_match/cache_controller.h>
#include <ash/memory/portable_buddy_system.h>
#include <queue>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <condition_variable>
#include <optional>

namespace cu_match {

class device_manager {

public:
    struct config_t {
        query_graph* qgraph;
        uint64_t host_buffer_size;
        uint64_t device_buffer_size;
        gstream::grid_file_stream* gfs;
        bool is_table_join;
        bool in_memory_mode;
    };
    
    device_manager();
    ~device_manager() noexcept;
    void init(config_t const& cfg);
    bool execute_match(lgf_scheduler* lgf_sched);
    bool execute_match_with_double_buffering(lgf_scheduler* lgf_sched);
    

private:
    config_t _cfg;
    uint32_t _num_query_vertices;
    uint32_t _num_query_edges;
    uint32_t _num_constraints;
    uint32_t _num_max_negative_vertices;
    int _num_gpus;
    qgraph_vertex_info* _vertices_info;
    qgraph_edge_info* _edges_info;
    global_kernel_task_info* _gk_info;
    kernel_executor* _kexec;
    uint64_t _aligned_hostbuf_size;
    uint64_t _aligned_devbuf_size;
    uint64_t _output_offset;
    uint64_t _max_header_size;
    void* _host_header_buffer[MaxNumGPUs];
    void* _device_buffer[MaxNumGPUs];
    std::queue<single_kernel_task_info*> _task_queue;
    std::vector<std::vector<sleaf_ptr>* > _sleaf_vecs;
    std::mutex _queue_mutex; // for task queue
    std::mutex _load_mutex; // for multi-GPU
    std::vector<sleaf_ptr> _shard_list; // for multi-GPU
    std::condition_variable _cv; // for multi-GPU
    std::atomic<bool> _ready; // for multi-GPU
    std::unordered_map<gstream::grid_format::shard_uid, uint32_t> _shard_mapper[MaxNumGPUs];

    struct memory_management {
        ash::portable_buddy_system* host_buddy;
        shard_cachectl* shard_cctl;

        std::function<void*(uint64_t size)> malloc_header_buffer;
        std::function<void*(sleaf_ptr const& sleaf)> malloc_shard_buffer;
        std::function<void(void*)> free_buffer;
        std::function<bool()> acquire_memory_space;
    } _mmgmt {};


    std::optional<single_kernel_task_info*> pop_queue_front() {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        if (!_task_queue.empty()) {
            single_kernel_task_info* ret = _task_queue.front();
            _task_queue.pop();
            return ret;
        }
        return std::nullopt;
    }

    uint64_t _make_header_buffer(single_kernel_task_info* k_info, uint32_t const g_id, uint32_t const stream_id);
    uint64_t _get_max_header_size();
    uint64_t _make_data_buffer_with_H2D(single_kernel_task_info* k_info, uint64_t const start_offset, gstream::cuda::stream_type const& stream, uint32_t const g_id, uint32_t const stream_id);
    void _kernel_H2D(uint64_t size, gstream::cuda::stream_type const& stream,  uint32_t const g_id, uint32_t const stream_id);
    void _kernel_D2H(uint64_t& output, gstream::cuda::stream_type const& stream,  uint32_t const g_id, uint32_t const stream_id);
    void _make_shard_list();
    void _graph_load_func();

    void* _allocate_shard_buffer_with_evcit(sleaf_ptr const& sleaf) const;
    void* _allocate_shard_buffer_with_evcit_for_multi(sleaf_ptr const& sleaf) const; // for multi-GPU
    bool  _evict_cached_shard_once() const;
    void  _free_shard_buffer(void*) const;
    void  _clear_cache() const;

    void _print_kernel_task_info(single_kernel_task_info* k_info);
    void _cleanup();
};

} // namespace cu_match

#endif // CU_MATCH_DEVICE_MANAGER_CUH
