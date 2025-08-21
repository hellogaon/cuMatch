#ifndef CU_MATCH_QUERY_GRAPH_H
#define CU_MATCH_QUERY_GRAPH_H
#include <cu_match/subgraph_match_defines.h>
#include <gstream/grid_format/grid_file_stream.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>

namespace cu_match {

class query_graph {

public:
    struct config_t {
        uint32_t num_vertices;
        uint32_t num_edges;
        uint32_t num_constraints;
        parse_vertex_info* parse_vertices_info;
        parse_edge_info* parse_edges_info;
        parse_constraint_info* parse_constraints_info;
        gstream::grid_file_stream* gfs;
    };
    
    query_graph();
    ~query_graph() noexcept;
    bool make_qgraph(config_t const& cfg);
    
    uint32_t num_query_vertices() const {
        return _cfg.num_vertices;
    }

    uint32_t num_query_edges() const {
        return _cfg.num_edges;
    }

    uint32_t num_constraints() const {
        return _cfg.num_constraints;
    }

    qgraph_vertex_info* qgraph_vertices_info() const {
        return _vertices_info;
    }

    qgraph_edge_info* qgraph_edges_info() const {
        return _edges_info;
    }

    qgraph_constraint_info* qgraph_constraints_info() const {
        return _constraints_info;
    }

    bool is_optional_pattern() const {
        for (uint32_t i = 0; i < _cfg.num_edges; i++)
            if (_edges_info[i].e_type == qgraph_edge_type::OPTIONAL) {
                return true;
            }
        return false;
    }

    bool is_negative_pattern() const {
        for (uint32_t i = 0; i < _cfg.num_edges; i++)
            if (_edges_info[i].e_type == qgraph_edge_type::NEGATIVE) {
                return true;
            }
        return false;
    }

    uint32_t num_optional_vertices() const {
        uint32_t counter = 0;
        for (uint32_t i = 0; i < _cfg.num_vertices; i++)
            if(_vertices_info[i].is_optional) 
                counter += 1;
        return counter;
    }

    uint32_t num_max_negative_vertices() const {
        uint32_t counter = 0;
        for (uint32_t i = 0; i < _cfg.num_edges; i++)
            if(_edges_info[i].e_type == qgraph_edge_type::NEGATIVE) 
                counter += 1;
        return counter;
    }

private:
    config_t _cfg;
    qgraph_vertex_info* _vertices_info;
    qgraph_edge_info* _edges_info;
    qgraph_constraint_info* _constraints_info;
    std::map<std::string, gstream::grid_off> _vname_map;

    bool _init(config_t const& cfg);
    void _cleanup();
    bool _make_query_graph();
    void _print_qgraph();
};

} // namespace cu_match

#endif // CU_MATCH_QUERY_GRAPH_H