#include <cu_match/device_defines.cuh>

namespace cu_match {

GSTREAM_DEVICE_ONLY 
void print_kernel_buffer(kernel_buffer_header const* kbuf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\theader_size: %lu\n", kbuf->header_size);
        printf("\tnum_query_vertices: %u\n", kbuf->num_query_vertices);
        printf("\tnum_query_edges: %u\n", kbuf->num_query_edges);
        printf("\tnum_constraints: %u\n", kbuf->num_constraints);
        printf("\tnum_subtasks: %u\n", kbuf->num_subtasks);
        printf("\tnum_sched_tree_nodes: %u\n", kbuf->num_sched_tree_nodes);
        printf("\tnum_slices: %u\n", kbuf->num_shards);
        printf("\tglobal_order: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_vertices; j++) {
            printf("%u ", kbuf->global_order[j]);
        }
        printf("]\n");
        printf("\tprefix_sum_degree: [ ");
        for (uint32_t i = 0; i < kbuf->num_query_vertices + 1; i++)
            printf("%u ", kbuf->prefix_sum_degree[i]);
        printf("]\n");
        printf("\teid_list: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_edges * 2; j++) {
            printf("%u ", kbuf->eid_list[j]);
        }
        printf("]\n");
        printf("\tedge_type_list: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_edges * 2; j++) {
            printf("%u ", kbuf->edge_type_list[j]);
        }
        printf("]\n");
        printf("\tlv0_dep: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_edges; j++) {
            printf("%u ", kbuf->lv0_dep[j]);
        }
        printf("]\n");
        printf("\tconstraint_ptr: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_vertices + 1; j++) {
            printf("%u ", kbuf->constraint_ptr[j]);
        }
        printf("]\n");
        printf("\tconstraints: [ ");
        for (uint32_t j = 0; j < kbuf->num_constraints; j++) {
            printf("%u ", kbuf->constraints[j]);
        }
        printf("]\n");
        printf("\tnum_optional_per_vertex: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_vertices; j++) {
            printf("%u ", kbuf->num_optional_per_vertex[j]);
        }
        printf("]\n");
        printf("\tnum_negative_per_vertex: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_vertices; j++) {
            printf("%u ", kbuf->num_negative_per_vertex[j]);
        }
        printf("]\n");
        printf("\tis_optional_vertex: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_vertices; j++) {
            printf("%u ", kbuf->is_optional_vertex[j]);
        }
        printf("]\n");
        printf("\tis_negative_vertex_with_degree: [ ");
        for (uint32_t j = 0; j < kbuf->num_query_vertices; j++) {
            printf("%u ", kbuf->is_negative_vertex_with_degree[j]);
        }
        printf("]\n");
        if (kbuf->is_table_join) {
            printf("\tedge_order_with_vid: [ ");
            for (uint32_t j = 0; j < kbuf->num_query_edges * 2; j++) {
                printf("%u ", kbuf->edge_order_with_vid[j]);
            }
            printf("]\n");
            printf("\tis_join_vertex: [ ");
            for (uint32_t j = 0; j < kbuf->num_query_edges * 2; j++) {
                printf("%u ", kbuf->is_join_vertex[j]);
            }
            printf("]\n");
        }
        printf("\tsched_tree:\n");
        printf("\t\tsrc: [ ");
        for (uint32_t i = 0; i < kbuf->num_sched_tree_nodes; i++)
            printf("%u ", i);
        printf("]\n");
        printf("\t\tptr: [ ");
        for (uint32_t i = 0; i < kbuf->num_sched_tree_nodes + 1; i++)
            printf("%u ", kbuf->sched_tree_ptr[i]);
        printf("]\n");
        printf("\t\tnxt id: [ ");
        for (uint32_t i = 0; i < kbuf->num_sched_tree_nodes - 1; i++)
            printf("%u ", kbuf->sched_tree_nxt_id[i]);
        printf("]\n");
        printf("\t\tprv id: [ ");
        for (uint32_t i = 0; i < kbuf->num_sched_tree_nodes - 1; i++)
            printf("%u ", kbuf->sched_tree_prv_id[i]);
        printf("]\n");
        printf("\t\tsubtask id: [ ");
        for (uint32_t i = 0; i < kbuf->num_sched_tree_nodes - 1; i++)
            printf("%u ", kbuf->sched_tree_st_id[i]);
        printf("]\n");
        printf("\tshard_info : [\n");
        for (uint32_t i = 0; i < kbuf->num_shards; i++) {
            using namespace gstream;
            flip24_shard_ptr shard = kbuf->shard_ptr[i];
            gbid_t gbid = shard->gbid();
            printf("\t\tshard %u (%u, %u) \n", i, gbid.row, gbid.col);
            printf("\t\t\tedges: [ ");
            uint32_t const num_rows = shard->num_lists();
            for (uint32_t j = 0; j < min(num_rows, 2u); j++) {
                flip24_shard::adj_list_t const adj = shard->adj_list(j);
                uint32_t src = adj.src_vertex;
                flip24_element const* col = shard->colv();
                for (uint32_t k = 0; k < min(adj.length, 2u); k++) {
                    uint32_t dst = read_flip24_element(col, k);
                    printf("(%u %u) ", src, dst);
                }
            }
            printf("... ]\n");
        }
        printf("\t]\n\n");
        
        for (uint32_t i = 0; i < kbuf->num_subtasks; i++) {
            printf("\tsubtask %u:\n", i);
            printf("\t\tshard_idx_list: [ ");
            for (uint32_t j = 0; j < kbuf->num_query_edges * 2; j++) {
                printf("%u ", kbuf->subtask_info[i].shard_idx_list[j]);
            }
            printf("]\n");
        }

        printf("\n");
    }
}

GSTREAM_DEVICE_ONLY
bool get_bit(uint32_t const vertex_id, uint32_t const* bitmap) {
    return bitmap[vertex_id >> 5UL] & (1 << (vertex_id & 31));
}

GSTREAM_DEVICE_ONLY
void set_bit(uint32_t const vertex_id, uint32_t* bitmap) {
    CUDA_atomicOr(
        &bitmap[vertex_id >> 5UL],
        1 << (vertex_id & 31)
    );
}

GSTREAM_DEVICE_ONLY
bool lower_bound_on_src_array(flip24_shard_ptr const& shard, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt) { // [start-end)
    // uint32_t lo = start, hi = start + 1;
    // uint32_t scale = 8;
    // while (hi < end && shard->source_vertex(hi) <= key) {
    //     lo = hi;
    //     hi += scale;
    //     scale <<= 3;
    // }
    // if (hi > end) hi = end;
    uint32_t lo = start, hi = end;

    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (key >= shard->source_vertex(mid))
            lo = mid;
        else
            hi = mid;
    }
    rslt = lo;
    return (key == shard->source_vertex(lo));
}

GSTREAM_DEVICE_ONLY
bool lower_bound_on_dst_array(flip24_shard::adj_list_t const& adj, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt) { // [start-end)
    // uint32_t lo = start, hi = start + 1;
    // uint32_t scale = 8;
    // while (hi < end && adj[hi] <= key) {
    //     lo = hi;
    //     hi += scale;
    //     scale <<= 3;
    // }
    // if (hi > end) hi = end;
    uint32_t lo = start, hi = end;

    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (key >= adj[mid])
            lo = mid;
        else
            hi = mid;
    }
    rslt = lo;
    return (key == adj[lo]);
}

GSTREAM_DEVICE_ONLY
bool upper_bound_on_src_array(flip24_shard_ptr const& shard, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt) { // [start-end)
    // int32_t lo = start, hi = start + 1;
    // int32_t scale = 8;
    // while (hi < end && shard->source_vertex(hi) <= key) {
    //     lo = hi;
    //     hi += scale;
    //     scale <<= 3;
    // }
    // lo--; hi++;
    // if (hi + 1 > end) hi = end - 1;
    int32_t lo = start - 1, hi = end - 1;

    while (lo + 1 < hi) {
        int32_t mid = (lo + hi) / 2;
        if (key <= shard->source_vertex(mid))
            hi = mid;
        else
            lo = mid;
    }
    rslt = hi;
    return (key == shard->source_vertex(hi));
}

GSTREAM_DEVICE_ONLY
bool upper_bound_on_dst_array(flip24_shard::adj_list_t const& adj, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt) { // [start-end)
    // int32_t lo = start, hi = start + 1;
    // int32_t scale = 8;
    // while (hi < end && adj[hi] <= key) {
    //     lo = hi;
    //     hi += scale;
    //     scale <<= 3;
    // }
    // lo--; hi++;
    // if (hi + 1 > end) hi = end - 1;
    int32_t lo = start - 1, hi = end - 1;

    while (lo + 1 < hi) {
        int32_t mid = (lo + hi) / 2;
        if (key <= adj[mid])
            hi = mid;
        else
            lo = mid;
    }
    rslt = hi;
    
    return (key == adj[hi]);
}

GSTREAM_DEVICE_ONLY
bool lower_bound_on_flip24_array(gstream::flip24_element const* arr, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt) { // [start-end)
    // uint32_t lo = start, hi = start + 1;
    // uint32_t scale = 8;
    // while (hi < end && read_flip24_element(arr, hi) <= key) {
    //     lo = hi;
    //     hi += scale;
    //     scale <<= 3;
    // }
    // if (hi > end) hi = end;
    uint32_t lo = start, hi = end;

    while (lo + 1 < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (key >= read_flip24_element(arr, mid))
            lo = mid;
        else
            hi = mid;
    }
    rslt = lo;
    return (key == read_flip24_element(arr, lo));
}

GSTREAM_DEVICE_ONLY
bool upper_bound_on_flip24_array(gstream::flip24_element const* arr, uint32_t key, uint32_t start, uint32_t end, uint32_t& rslt) { // [start-end)
    // int32_t lo = start, hi = start + 1;
    // int32_t scale = 8;
    // while (hi < end && read_flip24_element(arr, hi) <= key) {
    //     lo = hi;
    //     hi += scale;
    //     scale <<= 3;
    // }
    // lo--; hi++;
    // if (hi + 1 > end) hi = end - 1;
    int32_t lo = start - 1, hi = end - 1;

    while (lo + 1 < hi) {
        int32_t mid = (lo + hi) / 2;
        if (key <= read_flip24_element(arr, mid))
            hi = mid;
        else
            lo = mid;
    }
    rslt = hi;
    return (key == read_flip24_element(arr, hi));
}

} // namespace cu_match