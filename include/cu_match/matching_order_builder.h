#ifndef CU_MATCH_MATCH_ORDER_BUILDER_H
#define CU_MATCH_MATCH_ORDER_BUILDER_H
#include <cu_match/subgraph_match_defines.h>
#include <cu_match/query_graph.h>

namespace cu_match {

class matching_order_builder {

public:
    struct config_t {
        query_graph* qgraph;
        bool is_table_join;
    };
    
    matching_order_builder();
    ~matching_order_builder() noexcept;
    void init(config_t const& cfg);
    bool gen_matching_order(); 

    order_element* order() const {
        return _order;
    }

    order_element* edge_order() const {
        return _edge_order;
    }

private:
    config_t _cfg;
    uint32_t _num_vertices;
    uint32_t _num_edges;
    order_element* _order;
    order_element* _edge_order;
    qgraph_vertex_info* _vertices_info;
    qgraph_edge_info* _edges_info;
    bool _is_optional_pattern;

    void _cleanup();
    void _attribute_join_validation();
    void _attribute_join_print_order();
    void _table_join_print_order();
};

} // namespace cu_match

#endif // CU_MATCH_MATCH_ORDER_BUILDER_H
