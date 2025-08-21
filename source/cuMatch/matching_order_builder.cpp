#include <cu_match/matching_order_builder.h>

namespace cu_match {

matching_order_builder::matching_order_builder() {}

matching_order_builder::~matching_order_builder() {
    _cleanup();
}

void matching_order_builder::init(config_t const& cfg) {
    _cfg = cfg;
    _num_vertices = _cfg.qgraph->num_query_vertices();
    _num_edges = _cfg.qgraph->num_query_edges();
    _vertices_info = _cfg.qgraph->qgraph_vertices_info();
    _edges_info = _cfg.qgraph->qgraph_edges_info();
    _is_optional_pattern = _cfg.qgraph->is_optional_pattern();
    _order = static_cast<order_element*>(calloc(_num_vertices, sizeof(order_element)));
    _edge_order = static_cast<order_element*>(calloc(_num_edges, sizeof(order_element)));
}

bool matching_order_builder::gen_matching_order() {
    if (!_cfg.is_table_join) {
        for (uint32_t i = 0; i < _num_vertices; i++) { // TODO: matching order
            _order[i] = i;
            // _order[i] = _num_vertices - 1 - i;
        }
        _attribute_join_validation();
#ifdef DEBUG
        _attribute_join_print_order();
#endif

    }
    else {
        for (uint32_t i = 0; i < _num_edges; i++) { // TODO: matching order
            _edge_order[i] = i;
            // _edge_order[i] = _num_vertices - 1 - i;
        }
#ifdef DEBUG
        _table_join_print_order();
#endif
    }
    return true;
}

void matching_order_builder::_cleanup() {
    free(_order); _order = nullptr;
    free(_edge_order); _edge_order = nullptr;
}

void matching_order_builder::_attribute_join_validation() {
    if (_is_optional_pattern) { // for optional edge
        assert(_vertices_info[_order[0]].is_optional == false && "ordering error");
        
        for (uint32_t i = 1; i < _num_vertices; i++)
            assert(!(_vertices_info[_order[i - 1]].is_optional && !_vertices_info[_order[i]].is_optional) && "ordering error");

        // check topological order
        // uint32_t order_rank[MaxNumQueryVertices] = {};
        // for (uint32_t i = 0; i < _num_vertices; i++) {
        //     order_rank[_order[i]] = i;
        // }
        
        // struct inv_adj_info {
        //     uint32_t v_id;
        // };
        // std::vector<inv_adj_info> inv_adj[MaxNumQueryVertices];
        // for (uint32_t i = 0 ; i < _num_edges; i++) {
        //     uint32_t const v1_id = _edges_info[i].v1_id;
        //     uint32_t const v2_id = _edges_info[i].v2_id;
        //     uint32_t v1_rank = order_rank[v1_id];
        //     uint32_t v2_rank = order_rank[v2_id];
            
        //     if (v1_rank < v2_rank)
        //         inv_adj[v2_id].push_back( {v1_id } );
        //     else 
        //         inv_adj[v1_id].push_back( {v2_id } );
        // }

        // bool optional_visited[MaxNumQueryVertices] = {};
        // for (uint32_t i = 0; i < _num_vertices; i++) {
        //     uint32_t v_id = _order[i];
        //     if (_vertices_info[v_id].is_optional) {
        //         for (uint32_t j = 0; j < inv_adj[v_id].size(); j++) {
        //             assert(optional_visited[inv_adj[v_id][j].v_id] && "ordering error");
        //         }
                        
        //     }
        //     optional_visited[v_id] = true;
        // }
    }
}

void matching_order_builder::_attribute_join_print_order() {
    printf("> global match order:\n\t");
    for (uint32_t i = 0; i < _num_vertices; i++) {
        printf("%u (%s) ", _order[i], _vertices_info[_order[i]].name);
    }
    printf("\n");
    
}

void matching_order_builder::_table_join_print_order() {
    printf("> edge order:\n\t");
    for (uint32_t i = 0; i < _num_edges; i++) {
        printf("%u (%u-%u) ", _edge_order[i], _edges_info[_edge_order[i]].v1_id, _edges_info[_edge_order[i]].v2_id);
    }
    printf("\n");
    
}
} // namespace cu_match
