#ifndef CU_MATCH_SIMPLE_PARSER_H
#define CU_MATCH_SIMPLE_PARSER_H
#include <cu_match/subgraph_match_defines.h>
#include <gstream/grid_format/grid_file_stream.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace cu_match {

class simple_parser {

public:
    struct config_t {
        char const* query_file_path;
    };
    
    simple_parser();
    ~simple_parser() noexcept;
    bool parse(config_t const& cfg); 
    
    uint32_t num_query_vertices() const {
        return _num_query_vertices;
    }

    uint32_t num_query_edges() const {
        return _num_query_edges;
    }

    uint32_t num_constraints() const {
        return _num_constraints;
    }

    parse_vertex_info* parse_vertices_info() const {
        return _parse_vertices_info;
    }

    parse_edge_info* parse_edges_info() const {
        return _parse_edges_info;
    }

    parse_constraint_info* parse_constraints_info() const {
        return _parse_constraints_info;
    }

private:
    config_t _cfg;

    bool _init(config_t const& cfg);
    void _cleanup();
    bool _parse_json();
    bool _parse_query_vertices_data(nlohmann::json& data);
    bool _parse_query_edges_data(nlohmann::json& data);
    bool _parse_constraints_data(nlohmann::json& data);
    void _print_parse_infos();

    uint32_t _num_query_vertices;
    uint32_t _num_query_edges;
    uint32_t _num_constraints;
    parse_vertex_info* _parse_vertices_info;
    parse_edge_info* _parse_edges_info;
    parse_constraint_info* _parse_constraints_info;

    std::vector<std::string> _edge_type_tokens = {"OPTIONAL", "NOT"};
    std::vector<std::string> _constraint_tokens = {"<>"};
};

} // namespace cu_match

#endif // CU_MATCH_SIMPLE_PARSER_H
