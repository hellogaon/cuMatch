#include <cu_match/query_graph.h>

namespace cu_match {

query_graph::query_graph() {}

query_graph::~query_graph() {
    _cleanup();
}

bool query_graph::make_qgraph(config_t const& cfg) {
    if (!_init(cfg))
        return false;
    if (!_make_query_graph())
        return false;
#ifdef DEBUG
    _print_qgraph();
#endif
    return true;
}

bool query_graph::_init(config_t const& cfg) {
    _cfg = cfg;
    _vertices_info = static_cast<qgraph_vertex_info*>(calloc(_cfg.num_vertices, sizeof(qgraph_vertex_info)));
    _edges_info = static_cast<qgraph_edge_info*>(calloc(_cfg.num_edges, sizeof(qgraph_edge_info)));
    _constraints_info = static_cast<qgraph_constraint_info*>(calloc(_cfg.num_constraints, sizeof(qgraph_constraint_info)));
    return true;
}

void query_graph::_cleanup() {
    free(_vertices_info); _vertices_info = nullptr;
    free(_edges_info); _edges_info = nullptr;
    free(_constraints_info); _constraints_info = nullptr;
}

bool query_graph::_make_query_graph() {
    assert(_cfg.num_vertices > 0);
    assert(_cfg.num_edges > 0);
    for (uint32_t i = 0; i < _cfg.num_vertices; i++) {
        std::string label = _cfg.parse_vertices_info[i].label;
        std::string name = _cfg.parse_vertices_info[i].name;

        assert(_cfg.gfs->has_vertex_label(label));
        assert(_vname_map.find(name) == _vname_map.end());

        gstream::grid_off vl_id = _cfg.gfs->get_vertex_label_id(label);

        _vertices_info[i].vl_id = vl_id;
        strcpy(_vertices_info[i].name, name.c_str());
        _vertices_info[i].degree = 0;
        _vertices_info[i].is_optional = false;
        _vname_map[name] = i;
    }
    
    for (uint32_t i = 0; i < _cfg.num_edges; i++) {
        std::string label = _cfg.parse_edges_info[i].label;
        std::string v1_name = _cfg.parse_edges_info[i].v1_name;
        std::string v2_name = _cfg.parse_edges_info[i].v2_name;
        qgraph_edge_type e_ty = _cfg.parse_edges_info[i].e_type;
        qgraph_edge_direction e_dir = _cfg.parse_edges_info[i].e_dir;
        
        assert(_cfg.gfs->has_edge_label(label));
        assert(_vname_map.find(v1_name) != _vname_map.end());
        assert(_vname_map.find(v2_name) != _vname_map.end());

        gstream::grid_off el_id = _cfg.gfs->get_edge_label_id(label);
        gstream::grid_off v1_id = _vname_map[v1_name];
        gstream::grid_off v2_id = _vname_map[v2_name];

        _edges_info[i].el_id = el_id;
        _edges_info[i].v1_id = v1_id;
        _edges_info[i].v2_id = v2_id;
        _edges_info[i].e_type = e_ty;
        _edges_info[i].e_dir = e_dir;
        _vertices_info[v1_id].degree++;
        _vertices_info[v2_id].degree++;

        assert(v1_id != v2_id);
        assert(_vertices_info[v1_id].degree <= MaxNumQueryDegrees);
        assert(_vertices_info[v2_id].degree <= MaxNumQueryDegrees);
    }

    assert((_edges_info[0].e_type != qgraph_edge_type::OPTIONAL) && "Optional edges must be written after other edges");
    for (uint32_t i = 0; i < _cfg.num_vertices; i++) {
        assert(_vertices_info[i].degree != 0 && "There are vertices with zero degree");
    }
    for (uint32_t i = 1; i < _cfg.num_edges; i++) {
        assert(!(_edges_info[i].e_type != qgraph_edge_type::OPTIONAL && _edges_info[i - 1].e_type == qgraph_edge_type::OPTIONAL) && "Optional edges must be written after other edges");
    }

    // optional info
    bool is_regular_vertex[MaxNumQueryVertices] = {};
    for (uint32_t i = 0; i < _cfg.num_edges; i++) {
        gstream::grid_off const v1_id = _edges_info[i].v1_id;
        gstream::grid_off const v2_id = _edges_info[i].v2_id;
        if (_edges_info[i].e_type == qgraph_edge_type::OPTIONAL) {
            assert(!(is_regular_vertex[v1_id] && is_regular_vertex[v2_id]) && "optional edge must be add new vertex");
            if (!is_regular_vertex[v1_id])
                _vertices_info[v1_id].is_optional = true;
            else if (!is_regular_vertex[v2_id])
                _vertices_info[v2_id].is_optional = true;
        }
        else {
            is_regular_vertex[v1_id] = true;
            is_regular_vertex[v2_id] = true;
        }
    }

    for (uint32_t i = 0; i < _cfg.num_edges; i++) {
        gstream::grid_off v1_id = _edges_info[i].v1_id;
        gstream::grid_off v2_id = _edges_info[i].v2_id;
        _edges_info[i].optional_edge_with_regular_vertex = (_edges_info[i].e_type == qgraph_edge_type::OPTIONAL && (_vertices_info[v1_id].is_optional ^ _vertices_info[v2_id].is_optional));
    }

    // negative info
    for (uint32_t i = 0; i < _cfg.num_edges; i++) {
        [[maybe_unused]] gstream::grid_off v1_id = _edges_info[i].v1_id;
        [[maybe_unused]] gstream::grid_off v2_id = _edges_info[i].v2_id;
        if (_edges_info[i].e_type == qgraph_edge_type::NEGATIVE) {
            assert(_vertices_info[v1_id].degree > 1 && _vertices_info[v2_id].degree > 1 && "Not supported yet");
        }
    }

    for (uint32_t i = 0; i < _cfg.num_constraints; i++) {
        std::string v1_name = _cfg.parse_constraints_info[i].v1_name;
        std::string v2_name = _cfg.parse_constraints_info[i].v2_name;
        qgraph_constraint_type c_ty = _cfg.parse_constraints_info[i].c_type;
        
        assert(_vname_map.find(v1_name) != _vname_map.end());
        assert(_vname_map.find(v2_name) != _vname_map.end());

        gstream::grid_off v1_id = _vname_map[v1_name];
        gstream::grid_off v2_id = _vname_map[v2_name];

        _constraints_info[i].v1_id = v1_id;
        _constraints_info[i].v2_id = v2_id;
        _constraints_info[i].c_type = c_ty;

        assert(_vertices_info[v1_id].vl_id == _vertices_info[v2_id].vl_id);
    }

    return true;
}

// print type
std::string print_qgraph_edge_type(qgraph_edge_type const& e_type) {
    switch (e_type) {
        case qgraph_edge_type::REGULAR:
            return "REGULAR";
        case qgraph_edge_type::OPTIONAL:
            return "OPTIONAL";
        case qgraph_edge_type::NEGATIVE:
            return "NEGATIVE";
        default:
            return "-";
    }
}
std::string print_qgraph_edge_direction(qgraph_edge_direction const& e_dir) {
    switch (e_dir) {
        case qgraph_edge_direction::DIRECTED:
            return "DIRECTED";
        case qgraph_edge_direction::UNDIRECTED:
            return "UNDIRECTED";
        default:
            return "-";
    }
}
std::string print_qgraph_constraint_type(qgraph_constraint_type const& c_type) {
    switch (c_type) {
        case qgraph_constraint_type::NOT_EQUAL:
            return "NOT_EQUAL";
        default:
            return "-";
    }
}

void query_graph::_print_qgraph() {
    printf("> qgraph_vertices_info:\n");
    for (uint32_t i = 0; i < _cfg.num_vertices; i++) {
        printf("\tv_id: %-3u vl_id: %3u:%-15s name: %s\n", i, _vertices_info[i].vl_id, _cfg.gfs->get_vertex_label(_vertices_info[i].vl_id).c_str(), _vertices_info[i].name);
    }
    printf("> qgraph_edges_info:\n");
    for (uint32_t i = 0; i < _cfg.num_edges; i++) {
        printf("\te_id: %-3u el_id: %3u:%-15s v1_id: %-3u v2_id: %-3u e_ty: %s  e_dir: %s\n", i, _edges_info[i].el_id, _cfg.gfs->get_edge_label(_edges_info[i].el_id).c_str(), _edges_info[i].v1_id, _edges_info[i].v2_id, print_qgraph_edge_type(_edges_info[i].e_type).c_str(), print_qgraph_edge_direction(_edges_info[i].e_dir).c_str());
    }
    printf("> constraints_info:\n");
    for (uint32_t i = 0; i < _cfg.num_constraints; i++) {
        printf("\tc_id: %-3u v1_id: %-3u v2_id: %-3u c_type: %s\n", i, _constraints_info[i].v1_id, _constraints_info[i].v2_id, print_qgraph_constraint_type(_constraints_info[i].c_type).c_str());
    }
}

} // namespace cu_match
