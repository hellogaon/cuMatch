#include <cu_match/device_defines.cuh>
#include <gstream/grid_format/flip24_shard.h>

namespace cu_match {

GSTREAM_DEVICE_ONLY ASH_FORCEINLINE
uint32_t set_intersection_initial_negative_constraint(
        uint32_t const& dep,
        uint32_t const& num_query_vertices,
        uint32_t const& num_query_edges,
        uint32_t const& match_vid,
        uint32_t const& num_join_shards,
        uint32_t const& num_constraints_per_block, // for constraint
        uint32_t const* shard_idxs,
        flip24_shard_ptr const* shard_ptrs,
        uint32_t const* eids,
        uint32_t const* lv0_dep,
        uint32_t const* prefix_sum_degree,
        uint32_t const* constraint_ptr, // for constraint
        uint32_t const* constraints, // for constraint
        uint32_t const* rslt,
        uint32_t const* is_negative_vertex_with_degree, // for negative
        kernel_edge_type const* edge_type_list, // for negative
        uint32_t* compare_values, // for constraint
        uint32_t* vid_gap,
        uint32_t* vid_gap_idx,
        uint32_t min_iter_per_shard[][MaxNumQueryEdges * 2],
        uint32_t max_iter_per_shard[][MaxNumQueryEdges * 2],
        uint32_t* im_size,
        uint32_t im_data[][SMThreshold], // shared memory buffer
        gstream::flip24_element** flip24_arrs,
        intersection_type* is_type) {

    // get pivot vertex (min length)
    for (uint32_t i = threadIdx.x; i < num_join_shards; i += blockDim.x) {
        uint32_t const shard_offset = prefix_sum_degree[match_vid] + i;
        uint32_t const e_id = eids[shard_offset];
        uint32_t const shard_idx = shard_idxs[shard_offset];
        flip24_shard_ptr const shard = shard_ptrs[shard_idx];
        if (e_id < num_query_edges) {
            flip24_arrs[e_id] = const_cast<gstream::flip24_element*>(shard->rowv());
            vid_gap[i] = shard->num_lists();
            min_iter_per_shard[dep][e_id] = 0;
            max_iter_per_shard[dep][e_id] = vid_gap[i] - 1;
        }
        else {
            flip24_shard::adj_list_t const adj = shard->adj_list_vid(rslt[lv0_dep[e_id - num_query_edges]]);
            flip24_arrs[e_id] = const_cast<gstream::flip24_element*>(adj._colv);
            vid_gap[i] = adj.length;
            min_iter_per_shard[dep][e_id] = adj._colptr;
            max_iter_per_shard[dep][e_id] = adj._colptr + vid_gap[i] - 1;
        }
        vid_gap_idx[i] = shard_offset;
    }
    __syncthreads();

    // reduction min
    // for (uint32_t i = (num_join_shards + 1) / 2; i > 0; i >>= 1) {
    //     if (threadIdx.x < i) {
    //         if (vid_gap[threadIdx.x] > vid_gap[threadIdx.x + i]) {
    //             vid_gap[threadIdx.x] = vid_gap[threadIdx.x + i];
    //             vid_gap_idx[threadIdx.x] = vid_gap_idx[threadIdx.x + i];
    //         }
    //     }
    //     __syncthreads();
    // }
    if (threadIdx.x == 0) {
        for (uint32_t i = 1; i < num_join_shards - is_negative_vertex_with_degree[dep]; i++) { // for negative
            if (vid_gap[0] > vid_gap[i]) {
                vid_gap[0] = vid_gap[i];
                vid_gap_idx[0] = vid_gap_idx[i];
            }
        }
    }
    __syncthreads();

    uint32_t const pivot_shard_offset = vid_gap_idx[0];
    uint32_t const pivot_e_id = eids[pivot_shard_offset];
    gstream::flip24_element* const pivot_flip24_arr = flip24_arrs[pivot_e_id];

    if (!num_constraints_per_block) {
        if (dep != 0 && num_join_shards == 1) {
            if (threadIdx.x == 0) {
                im_size[dep] = vid_gap[0];
                is_type[dep] = intersection_type::SINGLE;
            }
            return pivot_shard_offset;
        }
        else if (dep == num_query_vertices - 1) {
            if (threadIdx.x == 0) {
                im_size[dep] = 0;
                is_type[dep] = intersection_type::LAST;
            }
            __syncthreads();

            uint32_t tmp;
            bool intersect;
            for (uint32_t i = threadIdx.x + min_iter_per_shard[dep][pivot_e_id]; i <= max_iter_per_shard[dep][pivot_e_id]; i += blockDim.x) {
                uint32_t v_id = read_flip24_element(pivot_flip24_arr, i);

                intersect = true;
                for (uint32_t j = 0; j < num_join_shards; j++) {
                    uint32_t const shard_offset = prefix_sum_degree[match_vid] + j;
                    if (shard_offset == pivot_shard_offset) continue;
                    uint32_t const e_id = eids[shard_offset];
                    
                    if (edge_type_list[shard_offset] != kernel_edge_type::NEGATIVE) {
                        intersect &= lower_bound_on_flip24_array(
                            flip24_arrs[e_id], 
                            v_id, 
                            min_iter_per_shard[dep][e_id], 
                            max_iter_per_shard[dep][e_id] + 1, 
                            tmp
                        );
                    }
                    else { // for negative
                        if (min_iter_per_shard[dep][e_id] == max_iter_per_shard[dep][e_id] + 1) {
                            intersect = true;
                        }
                        else {
                            intersect &= ~lower_bound_on_flip24_array(
                                flip24_arrs[e_id], 
                                v_id, 
                                min_iter_per_shard[dep][e_id], 
                                max_iter_per_shard[dep][e_id] + 1, 
                                tmp
                            );
                        }
                    }

                    if (!intersect)
                        break;
                }
                if (intersect) {
                    uint32_t idx = atomicAdd(&im_size[dep], 1);
                    // im_data[dep][idx] = v_id;
                }
            }
        }
        else {
            if (threadIdx.x == 0) {
                im_size[dep] = 0;
                is_type[dep] = intersection_type::SHARED_MEMBUF;
            }

            if (dep == 0 && threadIdx.x == 0) {
                uint32_t gap = (vid_gap[0] + MaxBlocks - 1) / MaxBlocks;
                max_iter_per_shard[dep][pivot_e_id] = min(min_iter_per_shard[dep][pivot_e_id] + gap * (blockIdx.x + 1) - 1, max_iter_per_shard[dep][pivot_e_id]);
                min_iter_per_shard[dep][pivot_e_id] += gap * blockIdx.x;
            }

            __syncthreads();

            uint32_t tmp;
            bool intersect;
            uint32_t max_value = min(max_iter_per_shard[dep][pivot_e_id], min_iter_per_shard[dep][pivot_e_id] + SMThreshold - 1);
            for (uint32_t i = threadIdx.x + min_iter_per_shard[dep][pivot_e_id]; i <= max_value; i += blockDim.x) {
                uint32_t v_id = read_flip24_element(pivot_flip24_arr, i);

                intersect = true;
                for (uint32_t j = 0; j < num_join_shards; j++) {
                    uint32_t const shard_offset = prefix_sum_degree[match_vid] + j;
                    if (shard_offset == pivot_shard_offset) continue;
                    uint32_t const e_id = eids[shard_offset];

                    if (edge_type_list[shard_offset] != kernel_edge_type::NEGATIVE) {
                        intersect &= lower_bound_on_flip24_array(
                            flip24_arrs[e_id], 
                            v_id, 
                            min_iter_per_shard[dep][e_id], 
                            max_iter_per_shard[dep][e_id] + 1, 
                            tmp
                        );
                    }
                    else { // for negative
                        if (min_iter_per_shard[dep][e_id] == max_iter_per_shard[dep][e_id] + 1) {
                            intersect = true;
                        }
                        else {
                            intersect &= ~lower_bound_on_flip24_array(
                                flip24_arrs[e_id], 
                                v_id, 
                                min_iter_per_shard[dep][e_id], 
                                max_iter_per_shard[dep][e_id] + 1, 
                                tmp
                            );
                        }
                    }

                    if (!intersect)
                        break;
                }
                if (intersect) {
                    uint32_t idx = atomicAdd(&im_size[dep], 1);
                    im_data[dep][idx] = v_id;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
                min_iter_per_shard[dep][pivot_e_id] += SMThreshold;

            return pivot_shard_offset;
        }
    }
    else { // for constraint
        if (dep == num_query_vertices - 1) {
            for (uint32_t i = threadIdx.x; i < num_constraints_per_block; i++) {
                compare_values[i] = rslt[constraints[constraint_ptr[match_vid] + i]];
            }

            if (threadIdx.x == 0) {
                im_size[dep] = 0;
                is_type[dep] = intersection_type::LAST;
            }
            __syncthreads();

            uint32_t tmp;
            bool intersect;
            for (uint32_t i = threadIdx.x + min_iter_per_shard[dep][pivot_e_id]; i <= max_iter_per_shard[dep][pivot_e_id]; i += blockDim.x) {
                uint32_t v_id = read_flip24_element(pivot_flip24_arr, i);

                intersect = true;
                for (uint32_t j = 0; j < num_constraints_per_block; j++)
                    if (v_id == compare_values[j]) {
                        intersect = false;
                        break;
                    }
                if (!intersect) continue;

                for (uint32_t j = 0; j < num_join_shards; j++) {
                    uint32_t const shard_offset = prefix_sum_degree[match_vid] + j;
                    if (shard_offset == pivot_shard_offset) continue;
                    uint32_t const e_id = eids[shard_offset];

                    if (edge_type_list[shard_offset] != kernel_edge_type::NEGATIVE) {
                        intersect &= lower_bound_on_flip24_array(
                            flip24_arrs[e_id], 
                            v_id, 
                            min_iter_per_shard[dep][e_id], 
                            max_iter_per_shard[dep][e_id] + 1, 
                            tmp
                        );
                    }
                    else { // for negative
                        if (min_iter_per_shard[dep][e_id] == max_iter_per_shard[dep][e_id] + 1) {
                            intersect = true;
                        }
                        else {
                            intersect &= ~lower_bound_on_flip24_array(
                                flip24_arrs[e_id], 
                                v_id, 
                                min_iter_per_shard[dep][e_id], 
                                max_iter_per_shard[dep][e_id] + 1, 
                                tmp
                            );
                        }
                    }

                    if (!intersect)
                        break;
                }
                if (intersect) {
                    uint32_t idx = atomicAdd(&im_size[dep], 1);
                    // im_data[dep][idx] = v_id;
                }
            }
        }
        else {
            for (uint32_t i = threadIdx.x; i < num_constraints_per_block; i++) {
                compare_values[i] = rslt[constraints[constraint_ptr[match_vid] + i]];
            }

            if (threadIdx.x == 0) {
                im_size[dep] = 0;
                is_type[dep] = intersection_type::SHARED_MEMBUF;
            }

            if (dep == 0 && threadIdx.x == 0) {
                uint32_t gap = (vid_gap[0] + MaxBlocks - 1) / MaxBlocks;
                max_iter_per_shard[dep][pivot_e_id] = min(min_iter_per_shard[dep][pivot_e_id] + gap * (blockIdx.x + 1) - 1, max_iter_per_shard[dep][pivot_e_id]);
                min_iter_per_shard[dep][pivot_e_id] += gap * blockIdx.x;
            }

            __syncthreads();

            uint32_t tmp;
            bool intersect;
            uint32_t max_value = min(max_iter_per_shard[dep][pivot_e_id], min_iter_per_shard[dep][pivot_e_id] + SMThreshold - 1);
            for (uint32_t i = threadIdx.x + min_iter_per_shard[dep][pivot_e_id]; i <= max_value; i += blockDim.x) {
                uint32_t v_id = read_flip24_element(pivot_flip24_arr, i);

                intersect = true;
                for (uint32_t j = 0; j < num_constraints_per_block; j++)
                    if (v_id == compare_values[j]) {
                        intersect = false;
                        break;
                    }
                if (!intersect) continue;

                for (uint32_t j = 0; j < num_join_shards; j++) {
                    uint32_t const shard_offset = prefix_sum_degree[match_vid] + j;
                    if (shard_offset == pivot_shard_offset) continue;
                    uint32_t const e_id = eids[shard_offset];

                    if (edge_type_list[shard_offset] != kernel_edge_type::NEGATIVE) {
                        intersect &= lower_bound_on_flip24_array(
                            flip24_arrs[e_id], 
                            v_id, 
                            min_iter_per_shard[dep][e_id], 
                            max_iter_per_shard[dep][e_id] + 1, 
                            tmp
                        );
                    }
                    else { // for negative
                        if (min_iter_per_shard[dep][e_id] == max_iter_per_shard[dep][e_id] + 1) {
                            intersect = true;
                        }
                        else {
                            intersect &= ~lower_bound_on_flip24_array(
                                flip24_arrs[e_id], 
                                v_id, 
                                min_iter_per_shard[dep][e_id], 
                                max_iter_per_shard[dep][e_id] + 1, 
                                tmp
                            );
                        }
                    }

                    if (!intersect)
                        break;
                }
                if (intersect) {
                    uint32_t idx = atomicAdd(&im_size[dep], 1);
                    im_data[dep][idx] = v_id;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
                min_iter_per_shard[dep][pivot_e_id] += SMThreshold;

            return pivot_shard_offset;
        }
    }
}


GSTREAM_DEVICE_ONLY ASH_FORCEINLINE
uint32_t set_intersection_resume_negative_constraint(
        uint32_t const& dep,
        uint32_t const& match_vid,
        uint32_t const& num_join_shards,
        uint32_t const& num_constraints_per_block, // for constraint
        uint32_t const& pivot_shard_offset,
        uint32_t const* eids,
        uint32_t const* prefix_sum_degree,
        uint32_t const* constraint_ptr, // for constraint
        uint32_t const* constraints, // for constraint
        uint32_t const* rslt, // for constraint
        kernel_edge_type const* edge_type_list, // for negative
        uint32_t* compare_values, // for constraint
        uint32_t min_iter_per_shard[][MaxNumQueryEdges * 2],
        uint32_t max_iter_per_shard[][MaxNumQueryEdges * 2],
        uint32_t* im_size,
        uint32_t im_data[][SMThreshold], // shared memory buffer
        gstream::flip24_element** flip24_arrs) {
    
    if (!num_constraints_per_block) {
        if (threadIdx.x == 0) {
            im_size[dep] = 0;
        }
        __syncthreads();

        uint32_t const pivot_e_id = eids[pivot_shard_offset];
        gstream::flip24_element* const pivot_flip24_arr = flip24_arrs[pivot_e_id];

        uint32_t tmp;
        bool intersect;
        uint32_t max_value = min(max_iter_per_shard[dep][pivot_e_id], min_iter_per_shard[dep][pivot_e_id] + SMThreshold - 1);
        for (uint32_t i = threadIdx.x + min_iter_per_shard[dep][pivot_e_id]; i <= max_value; i += blockDim.x) {
            uint32_t v_id = read_flip24_element(pivot_flip24_arr, i);

            intersect = true;
            for (uint32_t j = 0; j < num_join_shards; j++) {
                uint32_t const shard_offset = prefix_sum_degree[match_vid] + j;
                if (shard_offset == pivot_shard_offset) continue;
                uint32_t const e_id = eids[shard_offset];

                if (edge_type_list[shard_offset] != kernel_edge_type::NEGATIVE) {
                    intersect &= lower_bound_on_flip24_array(
                        flip24_arrs[e_id], 
                        v_id, 
                        min_iter_per_shard[dep][e_id], 
                        max_iter_per_shard[dep][e_id] + 1, 
                        tmp
                    );
                }
                else {
                    if (min_iter_per_shard[dep][e_id] == max_iter_per_shard[dep][e_id] + 1) { // for negative
                        intersect = true;
                    }
                    else {
                        intersect &= ~lower_bound_on_flip24_array(
                            flip24_arrs[e_id], 
                            v_id, 
                            min_iter_per_shard[dep][e_id], 
                            max_iter_per_shard[dep][e_id] + 1, 
                            tmp
                        );
                    }
                }

                if (!intersect)
                    break;
            }
            if (intersect) {
                uint32_t idx = atomicAdd(&im_size[dep], 1);
                im_data[dep][idx] = v_id;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            min_iter_per_shard[dep][pivot_e_id] += SMThreshold;
    }
    else { // for constraint
        for (uint32_t i = threadIdx.x; i < num_constraints_per_block; i++) {
            compare_values[i] = rslt[constraints[constraint_ptr[match_vid] + i]];
        }
        if (threadIdx.x == 0) {
            im_size[dep] = 0;
        }
        __syncthreads();

        uint32_t const pivot_e_id = eids[pivot_shard_offset];
        gstream::flip24_element* const pivot_flip24_arr = flip24_arrs[pivot_e_id];

        uint32_t tmp;
        bool intersect;
        uint32_t max_value = min(max_iter_per_shard[dep][pivot_e_id], min_iter_per_shard[dep][pivot_e_id] + SMThreshold - 1);
        for (uint32_t i = threadIdx.x + min_iter_per_shard[dep][pivot_e_id]; i <= max_value; i += blockDim.x) {
            uint32_t v_id = read_flip24_element(pivot_flip24_arr, i);

            intersect = true;
            for (uint32_t j = 0; j < num_constraints_per_block; j++) // for constraint
                if (v_id == compare_values[j]) {
                    intersect = false;
                    break;
                }
            if (!intersect) continue;
            
            for (uint32_t j = 0; j < num_join_shards; j++) {
                uint32_t const shard_offset = prefix_sum_degree[match_vid] + j;
                if (shard_offset == pivot_shard_offset) continue;
                uint32_t const e_id = eids[shard_offset];

                bool rslt = lower_bound_on_flip24_array( // for negative
                                    flip24_arrs[e_id], 
                                    v_id, 
                                    min_iter_per_shard[dep][e_id], 
                                    max_iter_per_shard[dep][e_id] + 1, 
                                    tmp
                                );
                if (edge_type_list[shard_offset] != kernel_edge_type::NEGATIVE) intersect &= rslt;
                else intersect &= ~rslt; // for negative

                if (!intersect)
                    break;
            }
            if (intersect) {
                uint32_t idx = atomicAdd(&im_size[dep], 1);
                im_data[dep][idx] = v_id;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            min_iter_per_shard[dep][pivot_e_id] += SMThreshold;
    }
}

GSTREAM_CUDA_KERNEL
void LFTJ_kernel_negative_constraint(kernel_args const args) {
    __shared__ uint32_t num_query_vertices;
    __shared__ uint32_t num_query_edges;
    __shared__ uint32_t num_constraints; // for constraint
    __shared__ uint32_t num_sched_tree_nodes;
    __shared__ uint32_t subtask_id;
    __shared__ uint32_t tree_node_id;
    __shared__ uint32_t dep;
    __shared__ uint32_t num_join_shards;
    __shared__ uint32_t match_vid;
    __shared__ uint32_t num_constraints_per_block; // for constraint

    __shared__ uint32_t sched_tree_ptr[MaxNumTreeNodes + 1];
    __shared__ uint32_t sched_tree_nxt_id[MaxNumTreeNodes - 1];
    __shared__ uint32_t sched_tree_prv_id[MaxNumTreeNodes - 1];
    __shared__ uint32_t sched_tree_st_id[MaxNumTreeNodes - 1];
    __shared__ uint32_t sched_tree_child[MaxNumQueryVertices];

    __shared__ uint32_t global_order[MaxNumQueryVertices];
    __shared__ uint32_t num_join_shards_arr[MaxNumQueryVertices];
    __shared__ uint32_t prefix_sum_degree[MaxNumQueryVertices + 1];
    __shared__ uint32_t constraint_ptr[MaxNumQueryVertices + 1]; // for constraint
    __shared__ uint32_t num_constraints_per_block_arr[MaxNumQueryVertices]; // for constraint
    __shared__ uint32_t constraints[MaxNumConstraints]; // for constraint
    __shared__ uint32_t compare_values[MaxNumConstraints]; // for constraint
    __shared__ uint32_t rslt[MaxNumQueryVertices];
    __shared__ bool is_finish[MaxNumQueryVertices];
    __shared__ state_type state[MaxNumQueryVertices];
    __shared__ uint32_t shard_offset_arr[MaxNumQueryVertices];
    __shared__ uint32_t im_size[MaxNumQueryVertices];
    __shared__ uint32_t tmp_im_size[MaxNumQueryVertices];
    __shared__ intersection_type is_type[MaxNumQueryVertices];
    __shared__ uint32_t im_data[MaxNumQueryVertices][SMThreshold];
    __shared__ uint32_t is_negative_vertex_with_degree[MaxNumQueryVertices]; // for negative
    __shared__ uint32_t num_negative_per_vertex[MaxNumQueryVertices]; // for negative

    __shared__ flip24_shard_ptr shard_ptrs[MaxNumShardPerIter];
    __shared__ uint32_t shard_idxs[MaxNumQueryEdges * 2];
    __shared__ uint32_t eids[MaxNumQueryEdges * 2];
    __shared__ uint32_t lv0_dep[MaxNumQueryEdges];
    __shared__ gstream::flip24_element* flip24_arrs[MaxNumQueryEdges * 2];
    __shared__ uint32_t min_iter_per_shard[MaxNumQueryVertices][MaxNumQueryEdges * 2];
    __shared__ uint32_t max_iter_per_shard[MaxNumQueryVertices][MaxNumQueryEdges * 2];
    __shared__ kernel_edge_type edge_type_list[MaxNumQueryEdges * 2]; // for negative

    __shared__ uint32_t vid_gap[MaxNumQueryDegrees];
    __shared__ uint32_t vid_gap_idx[MaxNumQueryDegrees]; // for find minimum vid_gap's idx


    kernel_buffer_header* kbuf = static_cast<kernel_buffer_header*>(args.buffer);
    gstream::cuda_uint64_t thread_local_counter = 0;

    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     print_kernel_buffer(kbuf);

    // init
    if (threadIdx.x == 0) {
        num_query_vertices = kbuf->num_query_vertices;
        num_query_edges = kbuf->num_query_edges;
        num_constraints = kbuf->num_constraints;
        num_sched_tree_nodes = kbuf->num_sched_tree_nodes;
        dep = 0;
    }
    for (uint32_t i = threadIdx.x; i < kbuf->num_shards; i += blockDim.x) {
        shard_ptrs[i] = kbuf->shard_ptr[i];
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i <= num_query_vertices; i += blockDim.x) {
        if (i < num_query_vertices) {
            rslt[i] = 0;
            is_finish[i] = true;
            state[i] = state_type::INITIAL;
            shard_offset_arr[i] = 0;
            sched_tree_child[i] = 0;
            global_order[i] = kbuf->global_order[i];
            is_negative_vertex_with_degree[i] = kbuf->is_negative_vertex_with_degree[i];
            num_negative_per_vertex[i] = kbuf->num_negative_per_vertex[i];
        }
        prefix_sum_degree[i] = kbuf->prefix_sum_degree[i];
        constraint_ptr[i] = kbuf->constraint_ptr[i];
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < num_query_vertices; i += blockDim.x) {
        num_join_shards_arr[i] = prefix_sum_degree[i + 1] - prefix_sum_degree[i] - num_negative_per_vertex[i];
        num_constraints_per_block_arr[i] = constraint_ptr[i + 1] - constraint_ptr[i];
    }

    for (uint32_t i = threadIdx.x; i < 2 * num_query_edges; i += blockDim.x) {
        eids[i] = kbuf->eid_list[i];
        edge_type_list[i] = kbuf->edge_type_list[i];
        if (i < num_query_edges) {
            lv0_dep[i] = kbuf->lv0_dep[i];
        }
    }
    for (uint32_t i = threadIdx.x; i < num_constraints; i += blockDim.x) {
        constraints[i] = kbuf->constraints[i];
    }
    for (uint32_t i = threadIdx.x; i <= num_sched_tree_nodes; i += blockDim.x) {
        sched_tree_ptr[i] = kbuf->sched_tree_ptr[i];
        if (i < num_sched_tree_nodes - 1) {
            sched_tree_nxt_id[i] = kbuf->sched_tree_nxt_id[i];
            sched_tree_prv_id[i] = kbuf->sched_tree_prv_id[i];
            sched_tree_st_id[i] = kbuf->sched_tree_st_id[i];
        }
    }
    for (uint32_t i = threadIdx.x; i < MaxNumQueryDegrees; i += blockDim.x) {
        vid_gap[i] = MaxNumVerticesPerGrid;
    }
    __syncthreads();

    for (uint32_t st_idx = 0; st_idx < kbuf->num_merged_subtasks; st_idx++) {
        if (threadIdx.x == 0) {
            subtask_id = kbuf->sched_tree_st_id[st_idx];
            tree_node_id = kbuf->tree_root_ids[st_idx];
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < 2 * num_query_edges; i += blockDim.x) {
            shard_idxs[i] = kbuf->subtask_info[subtask_id].shard_idx_list[i];
        }
        __syncthreads();

        while (true) {
            if (threadIdx.x == 0) {
                match_vid = global_order[dep];
                num_join_shards = num_join_shards_arr[match_vid];
                num_constraints_per_block = num_constraints_per_block_arr[match_vid];
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("> dep: %u, match_vid: %u num_join_shards: %u (%u-%u)\n", dep, match_vid, num_join_shards, prefix_sum_degree[match_vid + 1], prefix_sum_degree[match_vid]);
                //     printf("rslt: ");
                //     for (uint32_t i = 0; i < dep; i++)
                //         printf("%u ", rslt[i]);
                //     printf("\n");
                // }
            }
            __syncthreads();

            if (dep == num_query_vertices - 1) {
                uint32_t const pivot_shard_offset = set_intersection_initial_negative_constraint(
                    dep,
                    num_query_vertices,
                    num_query_edges, 
                    match_vid,
                    num_join_shards,
                    num_constraints_per_block,
                    shard_idxs,
                    shard_ptrs,
                    eids,
                    lv0_dep,
                    prefix_sum_degree,
                    constraint_ptr,
                    constraints,
                    rslt,
                    is_negative_vertex_with_degree,
                    edge_type_list,
                    compare_values,
                    vid_gap,
                    vid_gap_idx,
                    min_iter_per_shard,
                    max_iter_per_shard,
                    im_size,
                    im_data,
                    flip24_arrs,
                    is_type
                );
                __syncthreads();

                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     for (uint32_t i = 0; i < im_size[dep]; i++)
                //         printf("%lu / B: %u %u %u %u\n", thread_local_counter + i, rslt[0], rslt[1], rslt[2], im_data[dep][i]);
                // }
                if (threadIdx.x == 0) {
                    thread_local_counter += im_size[dep];                
                    tree_node_id = sched_tree_prv_id[tree_node_id - 1];
                    dep--;
                }
            }
            else {
                uint32_t pivot_shard_offset = shard_offset_arr[dep];
                uint32_t pivot_e_id = eids[pivot_shard_offset];
                gstream::flip24_element* flip24_arr = flip24_arrs[pivot_e_id];
                if (is_finish[dep]) { 
                    if (state[dep] == state_type::INITIAL) {
                        // calculate new intersection
                        pivot_shard_offset = set_intersection_initial_negative_constraint(
                            dep,
                            num_query_vertices,
                            num_query_edges, 
                            match_vid,
                            num_join_shards,
                            num_constraints_per_block,
                            shard_idxs,
                            shard_ptrs,
                            eids,
                            lv0_dep,
                            prefix_sum_degree,
                            constraint_ptr,
                            constraints,
                            rslt,
                            is_negative_vertex_with_degree,
                            edge_type_list,
                            compare_values,
                            vid_gap,
                            vid_gap_idx,
                            min_iter_per_shard,
                            max_iter_per_shard,
                            im_size,
                            im_data,
                            flip24_arrs,
                            is_type
                        );
                        __syncthreads();

                        pivot_e_id = eids[pivot_shard_offset];
                        flip24_arr = flip24_arrs[pivot_e_id];
                        if (threadIdx.x == 0) {
                            shard_offset_arr[dep] = pivot_shard_offset;
                            is_finish[dep] = false;
                            if (is_type[dep] != intersection_type::SHARED_MEMBUF || min_iter_per_shard[dep][pivot_e_id] > max_iter_per_shard[dep][pivot_e_id])
                                state[dep] = state_type::INITIAL;
                            else 
                                state[dep] = state_type::RESUME;
                            tmp_im_size[dep] = im_size[dep];
                        }
                    }
                    else { // resume intersection
                        set_intersection_resume_negative_constraint(
                            dep,
                            match_vid,
                            num_join_shards,
                            num_constraints_per_block,
                            pivot_shard_offset,
                            eids,
                            prefix_sum_degree,
                            constraint_ptr,
                            constraints,
                            rslt,
                            edge_type_list,
                            compare_values,
                            min_iter_per_shard,
                            max_iter_per_shard,
                            im_size,
                            im_data,
                            flip24_arrs
                        );
                        __syncthreads();

                        if (threadIdx.x == 0) {
                            is_finish[dep] = false;
                            if (min_iter_per_shard[dep][pivot_e_id] > max_iter_per_shard[dep][pivot_e_id])
                                state[dep] = state_type::INITIAL;
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
                        for (uint32_t i = threadIdx.x; i < 2 * num_query_edges; i += blockDim.x) {
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
                    if (is_type[dep] == intersection_type::SHARED_MEMBUF) {
                        rslt[dep] = im_data[dep][--im_size[dep]];
                        tree_node_id = sched_tree_nxt_id[sched_tree_ptr[tree_node_id] + sched_tree_child[dep]];
                        dep++;
                    }
                    else if (is_type[dep] == intersection_type::SINGLE) {
                        if (!num_constraints_per_block) {
                            rslt[dep] = read_flip24_element(flip24_arr, min_iter_per_shard[dep][pivot_e_id] + (--im_size[dep]));
                        }
                        else { // for constraint
                            for (uint32_t i = 0; i < num_constraints_per_block; i++) {
                                compare_values[i] = rslt[constraints[constraint_ptr[match_vid] + i]];
                            }
                            while (true) {
                                rslt[dep] = read_flip24_element(flip24_arr, min_iter_per_shard[dep][pivot_e_id] + (--im_size[dep]));
                                for (uint32_t i = 0; i < num_constraints_per_block; i++) {
                                    if (compare_values[i] == rslt[dep]) continue;
                                }
                                break;
                            }
                        }
                        tree_node_id = sched_tree_nxt_id[sched_tree_ptr[tree_node_id] + sched_tree_child[dep]];
                        dep++;
                    }
                }
            }
            __syncthreads();
        }
    }

    atomicAdd(&(kbuf->counter), thread_local_counter);
}

} // namespace cu_match
