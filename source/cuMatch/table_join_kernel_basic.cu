#include <cu_match/device_defines.cuh>
#include <gstream/grid_format/flip24_shard.h>

namespace cu_match {

GSTREAM_DEVICE_ONLY ASH_FORCEINLINE
void MJ_intial_basic(
        uint32_t const& dep,
        uint32_t const& num_query_vertices,
        uint32_t const& num_query_edges,
        uint32_t const& shard_idx,
        flip24_shard_ptr const* shard_ptrs,
        uint32_t const* edge_order_with_vid,
        bool const* is_join_vertex,
        uint32_t* min_iter_source_vertex,
        uint32_t* min_iter_dest_vertex,
        uint32_t* max_iter_source_vertex,
        uint32_t* max_iter_dest_vertex,
        uint32_t* rslt,
        state_type* state,
        uint32_t* im_size,
        uint32_t im_data[][SMThresholdOpt],
        gstream::flip24_element** flip24_arrs) {

    flip24_shard_ptr const shard = shard_ptrs[shard_idx];
    if (threadIdx.x == 0) {
        im_size[dep] = 0;
        state[dep] = state_type::RESUME;
        flip24_arrs[dep * 2] = const_cast<gstream::flip24_element*>(shard->rowv());
        min_iter_source_vertex[dep] = 0;
        max_iter_source_vertex[dep] = shard->num_lists() - 1;
        if (is_join_vertex[dep * 2]) {
            if (!shard->valid_source_vertex(rslt[edge_order_with_vid[dep * 2]]))
                state[dep] = state_type::INITIAL;
        }
    }
    __syncthreads();

    if (state[dep] == state_type::INITIAL)
        return;

    if (dep == num_query_edges - 1) {
        if (threadIdx.x == 0) {
            uint32_t v1_id;
            if (!is_join_vertex[dep * 2]) {
                v1_id = read_flip24_element(flip24_arrs[dep * 2], min_iter_source_vertex[dep]);
                rslt[edge_order_with_vid[dep * 2]] = v1_id;
            }
            else
                v1_id = rslt[edge_order_with_vid[dep * 2]];
            flip24_shard::adj_list_t const adj = shard->adj_list_vid(v1_id);
            flip24_arrs[dep * 2 + 1] = const_cast<gstream::flip24_element*>(adj._colv);
            min_iter_dest_vertex[dep] = adj._colptr;
            max_iter_dest_vertex[dep] = adj._colptr + adj.length - 1;
        }
        __syncthreads();

        while (true) {
            for (uint32_t i = threadIdx.x + min_iter_dest_vertex[dep]; i <= max_iter_dest_vertex[dep]; i += blockDim.x) {
                uint32_t v2_id = read_flip24_element(flip24_arrs[dep * 2 + 1], i);
                if (!(is_join_vertex[dep * 2 + 1] && rslt[edge_order_with_vid[dep * 2 + 1]] != v2_id)) {
                    uint32_t idx = atomicAdd(&im_size[dep], 1);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                min_iter_source_vertex[dep]++;
            }
            __syncthreads();

            if (is_join_vertex[dep * 2] || min_iter_source_vertex[dep] > max_iter_source_vertex[dep]) {
                state[dep] = state_type::INITIAL;
                break;
            }

            if (threadIdx.x == 0) {
                uint32_t v1_id = read_flip24_element(flip24_arrs[dep * 2], min_iter_source_vertex[dep]);
                rslt[edge_order_with_vid[dep * 2]] = v1_id;
                
                flip24_shard_ptr const shard = shard_ptrs[shard_idx];
                flip24_shard::adj_list_t const adj = shard->adj_list_vid(v1_id);
                flip24_arrs[dep * 2 + 1] = const_cast<gstream::flip24_element*>(adj._colv);
                min_iter_dest_vertex[dep] = adj._colptr;
                max_iter_dest_vertex[dep] = adj._colptr + adj.length - 1;
            }
            __syncthreads();
        }
    }
    else {
        if (dep == 0) {
            if (threadIdx.x == 0) {
                uint32_t gap = (max_iter_source_vertex[dep] + MaxBlocks - 1) / MaxBlocks;
                max_iter_source_vertex[dep] = min(min_iter_source_vertex[dep] + gap * (blockIdx.x + 1) - 1, max_iter_source_vertex[dep]);
                min_iter_source_vertex[dep] += gap * blockIdx.x;
            }
            __syncthreads();

            if (max_iter_source_vertex[dep] < min_iter_source_vertex[dep]) {
                state[dep] = state_type::INITIAL;
                return;
            }
        }
        __syncthreads();
        
        
        if (threadIdx.x == 0) {
            uint32_t v1_id;
            if (!is_join_vertex[dep * 2]) {
                v1_id = read_flip24_element(flip24_arrs[dep * 2], min_iter_source_vertex[dep]);
                rslt[edge_order_with_vid[dep * 2]] = v1_id;
            }
            else
                v1_id = rslt[edge_order_with_vid[dep * 2]];
            flip24_shard::adj_list_t const adj = shard->adj_list_vid(v1_id);
            flip24_arrs[dep * 2 + 1] = const_cast<gstream::flip24_element*>(adj._colv);
            min_iter_dest_vertex[dep] = adj._colptr;
            max_iter_dest_vertex[dep] = adj._colptr + adj.length - 1;
            if (is_join_vertex[dep * 2] && adj.length <= SMThresholdOpt) {
                state[dep] = state_type::INITIAL;
            }
        }
        __syncthreads();

        uint32_t max_value = min(max_iter_dest_vertex[dep], min_iter_dest_vertex[dep] + SMThresholdOpt - 1);
        for (uint32_t i = threadIdx.x + min_iter_dest_vertex[dep]; i <= max_value; i += blockDim.x) {
            uint32_t v2_id = read_flip24_element(flip24_arrs[dep * 2 + 1], i);

            if (!(is_join_vertex[dep * 2 + 1] && rslt[edge_order_with_vid[dep * 2 + 1]] != v2_id)) {
                uint32_t idx = atomicAdd(&im_size[dep], 1);
                im_data[dep][idx] = v2_id;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
            min_iter_dest_vertex[dep] += SMThresholdOpt;
    }
}


GSTREAM_DEVICE_ONLY ASH_FORCEINLINE
uint32_t MJ_resume_basic(
        uint32_t const& dep,
        uint32_t const* edge_order_with_vid,
        bool const* is_join_vertex,
        uint32_t const& shard_idx,
        flip24_shard_ptr const* shard_ptrs,
        uint32_t* min_iter_source_vertex,
        uint32_t* min_iter_dest_vertex,
        uint32_t* max_iter_source_vertex,
        uint32_t* max_iter_dest_vertex,
        uint32_t* rslt,
        state_type* state,
        uint32_t* im_size,
        uint32_t im_data[][SMThresholdOpt], // shared memory buffer
        gstream::flip24_element** flip24_arrs) {
    
    if (threadIdx.x == 0) {
        im_size[dep] = 0;
        state[dep] = state_type::RESUME;
        if (max_iter_dest_vertex[dep] < min_iter_dest_vertex[dep]) {
            if (is_join_vertex[dep * 2])
                state[dep] = state_type::INITIAL;
            else  {
                min_iter_source_vertex[dep]++;
                if (min_iter_source_vertex[dep] > max_iter_source_vertex[dep]) {
                    state[dep] = state_type::INITIAL;
                }
                else {
                    uint32_t v1_id = read_flip24_element(flip24_arrs[dep * 2], min_iter_source_vertex[dep]);
                    rslt[edge_order_with_vid[dep * 2]] = v1_id;
                    
                    flip24_shard_ptr const shard = shard_ptrs[shard_idx];
                    flip24_shard::adj_list_t const adj = shard->adj_list_vid(v1_id);
                    flip24_arrs[dep * 2 + 1] = const_cast<gstream::flip24_element*>(adj._colv);
                    min_iter_dest_vertex[dep] = adj._colptr;
                    max_iter_dest_vertex[dep] = adj._colptr + adj.length - 1;
                }
            }
        }
    }
    __syncthreads();

    if (state[dep] == state_type::INITIAL)
        return;
    
    uint32_t max_value = min(max_iter_dest_vertex[dep], min_iter_dest_vertex[dep] + SMThresholdOpt - 1);
    for (uint32_t i = threadIdx.x + min_iter_dest_vertex[dep]; i <= max_value; i += blockDim.x) {
        uint32_t v2_id = read_flip24_element(flip24_arrs[dep * 2 + 1], i);

        if (!(is_join_vertex[dep * 2 + 1] && rslt[edge_order_with_vid[dep * 2 + 1]] != v2_id)) {
            uint32_t idx = atomicAdd(&im_size[dep], 1);
            im_data[dep][idx] = v2_id;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
        min_iter_dest_vertex[dep] += SMThresholdOpt;
}

GSTREAM_CUDA_KERNEL
void MJ_kernel_basic(kernel_args const args) {
    __shared__ uint32_t num_query_vertices;
    __shared__ uint32_t num_query_edges;
    __shared__ uint32_t num_sched_tree_nodes;
    __shared__ uint32_t subtask_id;
    __shared__ uint32_t tree_node_id;
    __shared__ uint32_t dep;

    __shared__ uint32_t sched_tree_ptr[MaxNumTreeNodes + 1];
    __shared__ uint32_t sched_tree_nxt_id[MaxNumTreeNodes - 1];
    __shared__ uint32_t sched_tree_prv_id[MaxNumTreeNodes - 1];
    __shared__ uint32_t sched_tree_st_id[MaxNumTreeNodes - 1];
    __shared__ uint32_t sched_tree_child[MaxNumQueryEdges];

    __shared__ uint32_t rslt[MaxNumQueryVertices];
    __shared__ bool is_finish[MaxNumQueryEdges];
    __shared__ state_type state[MaxNumQueryEdges];
    __shared__ uint32_t im_size[MaxNumQueryEdges];
    __shared__ uint32_t tmp_im_size[MaxNumQueryEdges];
    __shared__ uint32_t im_data[MaxNumQueryEdges][SMThresholdOpt];

    __shared__ flip24_shard_ptr shard_ptrs[MaxNumShardPerIter];
    __shared__ uint32_t shard_idxs[MaxNumQueryEdges];
    __shared__ gstream::flip24_element* flip24_arrs[MaxNumQueryEdges * 2];
    __shared__ uint32_t edge_order_with_vid[MaxNumQueryEdges * 2];
    __shared__ bool is_join_vertex[MaxNumQueryEdges * 2];
    __shared__ uint32_t min_iter_source_vertex[MaxNumQueryEdges];
    __shared__ uint32_t min_iter_dest_vertex[MaxNumQueryEdges];
    __shared__ uint32_t max_iter_source_vertex[MaxNumQueryEdges];
    __shared__ uint32_t max_iter_dest_vertex[MaxNumQueryEdges];

    kernel_buffer_header* kbuf = static_cast<kernel_buffer_header*>(args.buffer);
    gstream::cuda_uint64_t thread_local_counter = 0;

    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     print_kernel_buffer(kbuf);

    // init
    if (threadIdx.x == 0) {
        num_query_vertices = kbuf->num_query_vertices;
        num_query_edges = kbuf->num_query_edges;
        num_sched_tree_nodes = kbuf->num_sched_tree_nodes;
        dep = 0;
    }
    for (uint32_t i = threadIdx.x; i < kbuf->num_shards; i += blockDim.x) {
        shard_ptrs[i] = kbuf->shard_ptr[i];
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_query_vertices; i += blockDim.x) {
        rslt[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < 2 * num_query_edges; i += blockDim.x) {
        edge_order_with_vid[i] = kbuf->edge_order_with_vid[i];
        is_join_vertex[i] = kbuf->is_join_vertex[i];
        if (i < num_query_edges) {
            sched_tree_child[i] = 0;
            state[i] = state_type::INITIAL;
            is_finish[i] = true;
        }
    }
    for (uint32_t i = threadIdx.x; i <= num_sched_tree_nodes; i += blockDim.x) {
        sched_tree_ptr[i] = kbuf->sched_tree_ptr[i];
        if (i < num_sched_tree_nodes - 1) {
            sched_tree_nxt_id[i] = kbuf->sched_tree_nxt_id[i];
            sched_tree_prv_id[i] = kbuf->sched_tree_prv_id[i];
            sched_tree_st_id[i] = kbuf->sched_tree_st_id[i];
        }
    }
    __syncthreads();

    for (uint32_t st_idx = 0; st_idx < kbuf->num_merged_subtasks; st_idx++) {
        if (threadIdx.x == 0) {
            subtask_id = kbuf->sched_tree_st_id[st_idx];
            tree_node_id = kbuf->tree_root_ids[st_idx];
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < num_query_edges; i += blockDim.x) {
            shard_idxs[i] = kbuf->subtask_info[subtask_id].shard_idx_list[i];
        }
        __syncthreads();

        while (true) {
            // if (dep == 1 && rslt[0] == 973 && rslt[1] == 621 && threadIdx.x == 0) {
            //     printf("> dep: %u (%u)\n", dep, blockIdx.x);
            //     printf("rslt: ");
            //     for (uint32_t i = 0; i < num_query_vertices; i++)
            //         printf("%u ", rslt[i]);
            //     printf("\n");
            // }
            __syncthreads();
            if (dep == num_query_edges - 1) {
                MJ_intial_basic(
                    dep,
                    num_query_vertices,
                    num_query_edges,
                    shard_idxs[dep],
                    shard_ptrs,
                    edge_order_with_vid,
                    is_join_vertex,
                    min_iter_source_vertex,
                    min_iter_dest_vertex,
                    max_iter_source_vertex,
                    max_iter_dest_vertex,
                    rslt,
                    state,
                    im_size,
                    im_data,
                    flip24_arrs
                );
                __syncthreads();

                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     for (uint32_t i = 0; i < im_size[dep]; i++)
                //         printf("%lu / B: %u %u %u %u\n", thread_local_counter + i, rslt[0], rslt[1], rslt[2], rslt[3]);
                // }

                // if (threadIdx.x == 0)
                //     printf("%lu / B: %u %u %u %u\n", thread_local_counter, rslt[0], rslt[1], rslt[2], im_size[dep]);
                //     // for (uint32_t i = 0; i < im_size[dep]; i++)
                //     //     printf("%lu / B: %u %u %u %u\n", thread_local_counter + i, rslt[0], rslt[1], rslt[2], im_data[dep][i]);
                // }
                if (threadIdx.x == 0) {
                    thread_local_counter += im_size[dep];
                    tree_node_id = sched_tree_prv_id[tree_node_id - 1];
                    dep--;
                }
            }
            else {
                if (is_finish[dep]) { 
                    if (state[dep] == state_type::INITIAL) {
                        __syncthreads();
                        // calculate new intersection
                        MJ_intial_basic(
                            dep,
                            num_query_vertices,
                            num_query_edges,
                            shard_idxs[dep],
                            shard_ptrs,
                            edge_order_with_vid,
                            is_join_vertex,
                            min_iter_source_vertex,
                            min_iter_dest_vertex,
                            max_iter_source_vertex,
                            max_iter_dest_vertex,
                            rslt,
                            state,
                            im_size,
                            im_data,
                            flip24_arrs
                        );
                        __syncthreads();

                        if (threadIdx.x == 0) {
                            is_finish[dep] = false;
                            tmp_im_size[dep] = im_size[dep];
                        }
                    }
                    else { // resume intersection
                        __syncthreads();
                        MJ_resume_basic(
                            dep,
                            edge_order_with_vid,
                            is_join_vertex,
                            shard_idxs[dep],
                            shard_ptrs,
                            min_iter_source_vertex,
                            min_iter_dest_vertex,
                            max_iter_source_vertex,
                            max_iter_dest_vertex,
                            rslt,
                            state,
                            im_size,
                            im_data,
                            flip24_arrs
                        );
                        __syncthreads();

                        if (threadIdx.x == 0) {
                            is_finish[dep] = false;
                            tmp_im_size[dep] = im_size[dep];
                        }
                    }
                }

                if (im_size[dep] == 0) {
                    uint32_t const child_num = sched_tree_ptr[tree_node_id + 1] - sched_tree_ptr[tree_node_id];
                    __syncthreads();
                    if (1 < child_num) { // for remove duplicate (iterate grid group)
                        if (threadIdx.x == 0) {
                            sched_tree_child[dep] = (sched_tree_child[dep] + 1) % child_num;
                            subtask_id = sched_tree_st_id[sched_tree_ptr[tree_node_id] + sched_tree_child[dep]];
                            im_size[dep] = tmp_im_size[dep];
                        }
                        __syncthreads();
                        for (uint32_t i = threadIdx.x; i < num_query_edges; i += blockDim.x) {
                            shard_idxs[i] = kbuf->subtask_info[subtask_id].shard_idx_list[i];
                        }
                    }
                    if (sched_tree_child[dep] == 0) { // if iterate all grid group
                        if (state[dep] == state_type::INITIAL && dep == 0) {
                            is_finish[dep] = true;
                            break;
                        }
                        __syncthreads();
                        if (threadIdx.x == 0) {
                            is_finish[dep] = true;
                            if (state[dep] == state_type::INITIAL) {
                                tree_node_id = sched_tree_prv_id[tree_node_id - 1];
                                dep--;
                            }
                        }
                    }
                    continue;
                }
                __syncthreads();
                
                if (threadIdx.x == 0) {
                    rslt[edge_order_with_vid[dep * 2 + 1]] = im_data[dep][--im_size[dep]];
                    tree_node_id = sched_tree_nxt_id[sched_tree_ptr[tree_node_id] + sched_tree_child[dep]];
                    dep++;
                }
            }
            __syncthreads();
        }
    }

    atomicAdd(&(kbuf->counter), thread_local_counter);
}

} // namespace cu_match
