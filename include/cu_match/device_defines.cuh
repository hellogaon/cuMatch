#ifndef CU_MATCH_DEVICE_DEFINES_CUH
#define CU_MATCH_DEVICE_DEFINES_CUH
#include <cu_match/subgraph_match_defines.h>
#include <gstream/cuda_env.h>
#include <gstream/grid_format/flip24_shard.h>

namespace cu_match {

constexpr unsigned WarpSize = 32;
constexpr unsigned NumThreads = 128;
constexpr unsigned MaxBlocks = 3840;
// constexpr unsigned NumThreads = 32;
// constexpr unsigned MaxBlocks = 1;
constexpr unsigned NumThreadsOpt = 32;
constexpr unsigned MaxBlocksOpt = 5760;
constexpr unsigned SMThreshold = NumThreads;
constexpr unsigned SMThresholdOpt = NumThreadsOpt;
constexpr unsigned NumWarps = NumThreads * MaxBlocks / WarpSize;
constexpr unsigned ShardMemSize = 0;
constexpr unsigned CudaAlignment = 256;
constexpr unsigned DiskIOSectorSize = 4096;
constexpr unsigned MaxNumGPUs = 1;

using query_pattern_type = unsigned;
constexpr query_pattern_type NORMAL_PATTERN = 0x00;
constexpr query_pattern_type CONSTRAINT_PATTERN = 0x01 << 1;
constexpr query_pattern_type OPTIONAL_PATTERN = 0x01 << 2;
constexpr query_pattern_type NEGATIVE_PATTERN = 0x01 << 3;

struct kernel_args {
    void* buffer;
};

enum class join_algorithm_type : char {
    ATTRIBUTE_JOIN = 0, // leap-frog trie join
    TABLE_JOIN = 1 // multi-way sort join
};

enum class kernel_edge_type : char {
    REGULAR = 0,
    OPTIONAL = 1,
    NEGATIVE = 2
};

enum class intersection_type : char {
    SHARED_MEMBUF = 0,
    GPU_MEMBUF = 1,
    SINGLE = 2,
    BITMAP = 3,
    LAST = 4
};

enum class state_type : char {
    INITIAL = 0,
    RESUME = 1
};

struct kernel_subtask_info {
    uint32_t* shard_idx_list; // size: num_query_edges * 2
};

struct kernel_buffer_header {
    uint64_t header_size;
    uint32_t num_query_vertices;
    uint32_t num_query_edges;
    uint32_t num_constraints;
    uint32_t num_subtasks;
    uint32_t num_merged_subtasks;
    uint32_t num_sched_tree_nodes;
    uint32_t num_shards;
    bool is_table_join;
    gstream::cuda_uint64_t counter;
    
    uint32_t* global_order; // size: num_query_vertices
    uint32_t* prefix_sum_degree; // size: num_query_vertices + 1
    uint32_t* eid_list; // size: num_query_edges * 2
    kernel_edge_type* edge_type_list; // size: num_query_edges * 2
    uint32_t* lv0_dep; // size: num_query_edges
    uint32_t* sched_tree_ptr; // size: num_sched_tree_nodes + 1
    uint32_t* sched_tree_nxt_id; // size: num_sched_tree_nodes - 1
    uint32_t* sched_tree_prv_id; // size: num_sched_tree_nodes - 1
    uint32_t* sched_tree_st_id; // size: num_sched_tree_nodes - 1
    flip24_shard_ptr* shard_ptr; // size: num_shards
    uint32_t* constraint_ptr; // size: num_query_vertices + 1
    uint32_t* constraints; // size: num_constraints
    uint32_t* tree_root_ids; // size: num_merged_subtasks
    kernel_subtask_info* subtask_info; // size: num_subtasks
    uint32_t* num_optional_per_vertex; // size: num_query_vertices 
    uint32_t* num_negative_per_vertex; // size: num_query_vertices 
    bool* is_optional_vertex; // size: num_query_vertices
    uint32_t* is_negative_vertex_with_degree; // size: num_query_vertices

    // for table join
    uint32_t* edge_order_with_vid; // size: num_query_edges * 2
    bool* is_join_vertex; // size: num_query_edges * 2
};

// for global task information (to make kernel buffer)
struct global_kernel_task_info {
    uint32_t* global_order;
    uint32_t* prefix_sum_degree;
    uint32_t* eid_list;
    kernel_edge_type* edge_type_list;
    uint32_t* lv0_dep;
    uint32_t* v1_list_idxs;
    uint32_t* v2_list_idxs;
    uint32_t* constraint_ptr;
    uint32_t* constraints;
    uint32_t* num_optional_per_vertex;
    uint32_t* num_negative_per_vertex;
    bool* is_optional_vertex;
    uint32_t* is_negative_vertex_with_degree;

    // for table join
    uint32_t* edge_order_with_vid;
    bool* is_join_vertex;

    ~global_kernel_task_info() {
        free(global_order);
        free(prefix_sum_degree);
        free(eid_list);
        free(edge_type_list);
        free(lv0_dep);
        free(v1_list_idxs);
        free(v2_list_idxs);
        free(constraint_ptr);
        free(constraints);
        free(num_optional_per_vertex);
        free(num_negative_per_vertex);
        free(is_optional_vertex);
        free(is_negative_vertex_with_degree);
        free(edge_order_with_vid);
        free(is_join_vertex);
    }
};

// for sub-task information (to make kernel buffer)
struct subtask_info {
    sleaf_ptr* sleaf_ptrs;

    ~subtask_info() {
        free(sleaf_ptrs);
    }
};
struct single_kernel_task_info {
    uint32_t num_tree_nodes;
    uint32_t* sched_tree_ptr;
    uint32_t* sched_tree_nxt_id;
    uint32_t* sched_tree_prv_id;
    uint32_t* sched_tree_st_id;
    std::vector<uint32_t> tree_root_id;
    std::vector<subtask_info*> subtasks;
    std::vector<sleaf_ptr> total_sleaf_ptrs;

    ~single_kernel_task_info() {
        free(sched_tree_ptr);
        free(sched_tree_nxt_id);
        free(sched_tree_prv_id);
        free(sched_tree_st_id);
        for (subtask_info* st_info: subtasks)
            delete st_info;
    }
};

// attribute join kernel
GSTREAM_CUDA_KERNEL void LFTJ_kernel_basic(kernel_args const args);
GSTREAM_CUDA_KERNEL void LFTJ_kernel_constraint(kernel_args const args);
GSTREAM_CUDA_KERNEL void LFTJ_kernel_optional(kernel_args const args);
GSTREAM_CUDA_KERNEL void LFTJ_kernel_negative(kernel_args const args);
GSTREAM_CUDA_KERNEL void LFTJ_kernel_negative_constraint(kernel_args const args);
GSTREAM_CUDA_KERNEL void LFTJ_kernel_optional_constraint(kernel_args const args);

// table join kernel
GSTREAM_CUDA_KERNEL void MJ_kernel_basic(kernel_args const args);

GSTREAM_DEVICE_ONLY void print_kernel_buffer(kernel_buffer_header const* kbuf);
GSTREAM_DEVICE_ONLY bool get_bit(uint32_t const vertex_id, uint32_t const* bitmap);
GSTREAM_DEVICE_ONLY void set_bit(uint32_t const vertex_id, uint32_t* bitmap);
GSTREAM_DEVICE_ONLY bool lower_bound_on_src_array(flip24_shard_ptr const& shard, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt);
GSTREAM_DEVICE_ONLY bool lower_bound_on_dst_array(flip24_shard::adj_list_t const& adj, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt);
GSTREAM_DEVICE_ONLY bool upper_bound_on_src_array(flip24_shard_ptr const& shard, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt);
GSTREAM_DEVICE_ONLY bool upper_bound_on_dst_array(flip24_shard::adj_list_t const& adj, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt);
GSTREAM_DEVICE_ONLY bool lower_bound_on_flip24_array(gstream::flip24_element const* arr, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt);
GSTREAM_DEVICE_ONLY bool upper_bound_on_flip24_array(gstream::flip24_element const* arr, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt);

template <typename T>
GSTREAM_DEVICE_COMPATIBLE
T* _seek_pointer(T* p, uint64_t offset) {
    auto const p2 = (char*)p;
    return reinterpret_cast<T*>(p2 + offset);
}

template <typename T>
GSTREAM_DEVICE_COMPATIBLE
T* _rev_seek_pointer(T* p, uint64_t offset) {
    auto const p2 = (char*)p;
    return reinterpret_cast<T*>(p2 - offset);
}

} // namespace cu_match

#endif // CU_MATCH_DEVICE_DEFINES_CUH
