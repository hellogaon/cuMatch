#include <cu_match/simple_parser.h>
#include <boost/tokenizer.hpp>
#include <ash/utility/dbg_log.h>
#include <fstream>

namespace cu_match {

simple_parser::simple_parser() {}

simple_parser::~simple_parser() {
    _cleanup();
}

bool simple_parser::parse(config_t const& cfg) {
    if (!_init(cfg))
        return false;
    return _parse_json();
}

bool simple_parser::_init(config_t const& cfg) {
    _cfg = cfg;
    return true;
}

void simple_parser::_cleanup() {
    free(_parse_vertices_info); _parse_vertices_info = nullptr;
    free(_parse_edges_info); _parse_edges_info = nullptr;
    free(_parse_constraints_info); _parse_constraints_info = nullptr;
}

bool simple_parser::_parse_json() {
    using json = nlohmann::json;

    std::ifstream ifs(_cfg.query_file_path);
    if (!ifs.is_open()) {
        ASH_ERRLOG("Query file open error");
        return false;
    }
    
    json full_data = json::parse(ifs);
    {
        json data = full_data["query_vertices"];
        bool rslt = _parse_query_vertices_data(data);
        if (!rslt) {
            ASH_ERRLOG("Query vertices parsing error");
            return false;
        }
    }
    
    {
        json data = full_data["query_edges"];
        bool rslt = _parse_query_edges_data(data);
        if (!rslt) {
            ASH_ERRLOG("Query edges parsing error");
            return false;
        }
    }
    
    {
        json data = full_data["constraints"];
        bool rslt = _parse_constraints_data(data);
        if (!rslt) {
            ASH_ERRLOG("Constraints parsing error");
            return false;
        }
    }
    // _print_parse_infos();
    return true;
}

bool simple_parser::_parse_query_vertices_data(nlohmann::json& data) {
    using json = nlohmann::json;

    _num_query_vertices = data.size();
    _parse_vertices_info = static_cast<parse_vertex_info*>(calloc(_num_query_vertices, sizeof(parse_vertex_info)));

    uint32_t i = 0;
    for (json::iterator it = data.begin(); it != data.end(); it++) {
        if (!it.value().is_string()) return false;
        std::string str = it.value();

        boost::char_separator<char> sep(":", ":", boost::keep_empty_tokens);
        boost::tokenizer<boost::char_separator<char> > tok(str, sep);
        
        std::vector<std::string> token_list;
        for (auto beg = tok.begin(); beg != tok.end(); beg++) {
            token_list.push_back(*beg);
        }

        std::string name, label, target;
        if (token_list.size() == 3) {
            name = token_list[0];
            label = token_list[2];
            target = name + ":" + label;
        }
        else 
            return false;

        if (str != target)
            return false;

        assert(name.size() < NameSizeLimit);
        assert(label.size() < LabelSizeLimit);

        strcpy(_parse_vertices_info[i].name, name.c_str());
        strcpy(_parse_vertices_info[i].label, label.c_str());
        i++;
    }
    assert(i <= MaxNumQueryVertices);
    return true;
}

bool simple_parser::_parse_query_edges_data(nlohmann::json& data) {
    using json = nlohmann::json;

    _num_query_edges = data.size();
    _parse_edges_info = static_cast<parse_edge_info*>(calloc(_num_query_edges, sizeof(parse_edge_info)));

    uint32_t i = 0;
    for (json::iterator it = data.begin(); it != data.end(); it++) {
        if (!it.value().is_string()) return false;
        std::string str = it.value();        

        boost::char_separator<char> sep(" <->", "", boost::keep_empty_tokens);
        boost::tokenizer<boost::char_separator<char> > tok(str, sep);
        
        std::vector<std::string> token_list;
        for (auto beg = tok.begin(); beg != tok.end(); beg++) {
            token_list.push_back(*beg);
        }

        std::string v1_name, label, v2_name, target;
        qgraph_edge_type e_ty = qgraph_edge_type::REGULAR;
        qgraph_edge_direction e_dir;

        if (token_list.size() == 5) {
            for (uint32_t i = 0; i < _edge_type_tokens.size(); i++) {
                std::string& e_token = _edge_type_tokens[i];
                if (token_list[0] == e_token) {
                    e_ty = static_cast<qgraph_edge_type>(i);
                    target = e_token + " ";
                }
            }
            token_list.erase(token_list.begin());
        }

        if (token_list.size() == 4) { // DIRECTED
            if (token_list[2] == "") { // OUTGOING
                v1_name = token_list[0];
                label = token_list[1];
                v2_name = token_list[3];
                e_dir = qgraph_edge_direction::DIRECTED;
                target += v1_name + "-" + label + "->" + v2_name;
            }
            else if(token_list[1] == "") { // INGOING
                v1_name = token_list[3];
                label = token_list[2];
                v2_name = token_list[0];
                e_dir = qgraph_edge_direction::DIRECTED;
                target += v2_name + "<-" + label + "-" + v1_name;
            }
            else
                return false;
        }
        else if (token_list.size() == 3) { // UNDIRECTED
            v1_name = token_list[0];
            label = token_list[1];
            v2_name = token_list[2];
            e_dir = qgraph_edge_direction::UNDIRECTED;
            target = v1_name + "-" + label + "-" + v2_name;
        }
        else
            return false;

        if (str != target)
            return false;
        
        assert(label[0] == '[' && label[label.size() - 1] == ']');
        assert(v1_name[0] == '(' && v1_name[v1_name.size() - 1] == ')');
        assert(v2_name[0] == '(' && v2_name[v2_name.size() - 1] == ')');
        label = label.substr(1, label.size() - 2);
        v1_name = v1_name.substr(1, v1_name.size() - 2);
        v2_name = v2_name.substr(1, v2_name.size() - 2);
        assert(label.size() < LabelSizeLimit);
        assert(v1_name.size() < NameSizeLimit);
        assert(v2_name.size() < NameSizeLimit);
    
        strcpy(_parse_edges_info[i].label, label.c_str());
        strcpy(_parse_edges_info[i].v1_name, v1_name.c_str());
        strcpy(_parse_edges_info[i].v2_name, v2_name.c_str());
        _parse_edges_info[i].e_type = e_ty;
        _parse_edges_info[i].e_dir = e_dir;
        i++;
    }
    assert(i <= MaxNumQueryEdges);
    return true;
}

bool simple_parser::_parse_constraints_data(nlohmann::json& data) {
    using json = nlohmann::json;

    _num_constraints = data.size();
    _parse_constraints_info = static_cast<parse_constraint_info*>(calloc(_num_constraints, sizeof(parse_constraint_info)));

    uint32_t i = 0;
    for (json::iterator it = data.begin(); it != data.end(); it++) {
        if (!it.value().is_string()) return false;
        std::string str = it.value();        

        std::string found = "";

        for (uint32_t i = 0; i < _constraint_tokens.size(); i++) {
            std::string& c_token = _constraint_tokens[i];
            if (str.find(c_token) != std::string::npos) {
                found = c_token;
                break;
            }
        }
            
        if (found == "") return false;

        boost::char_separator<char> sep(found.c_str(), "", boost::keep_empty_tokens);
        boost::tokenizer<boost::char_separator<char> > tok(str, sep);

        std::vector<std::string> token_list;
        for (auto beg = tok.begin(); beg != tok.end(); beg++) {
            token_list.push_back(*beg);
        }

        std::string v1_name, v2_name, target;
        qgraph_constraint_type c_ty = qgraph_constraint_type::REGULAR;
        
        for (uint32_t i = 0; i < _constraint_tokens.size(); i++) {
            std::string& c_token = _constraint_tokens[i];
            if (found == c_token) {
                v1_name = token_list[0];
                v2_name = token_list[2];
                c_ty = static_cast<qgraph_constraint_type>(i);
                target = v1_name + c_token + v2_name;
                assert(v1_name != v2_name);
            }
        }
        if (str != target)
            return false;

        assert(v1_name.size() < NameSizeLimit);
        assert(v2_name.size() < NameSizeLimit);
    
        strcpy(_parse_constraints_info[i].v1_name, v1_name.c_str());
        strcpy(_parse_constraints_info[i].v2_name, v2_name.c_str());
        _parse_constraints_info[i].c_type = c_ty;
        i++;
    }
    assert(i <= MaxNumConstraints);
    return true;
}

void simple_parser::_print_parse_infos() {
    printf("> parse vertices info:\n");
    for (uint32_t i = 0; i < _num_query_vertices; i++) {
        printf("\tlabel: %-15s name: %-15s\n", _parse_vertices_info[i].label, _parse_vertices_info[i].name);
    }
    printf("> parse edges info:\n");
    for (uint32_t i = 0; i < _num_query_edges; i++) {
        printf("\tlabel: %-15s name1: %-15s name2: %-15s\n", _parse_edges_info[i].label, _parse_edges_info[i].v1_name, _parse_edges_info[i].v2_name);
    }
    printf("> constraints info:\n");
    for (uint32_t i = 0; i < _num_constraints; i++) {
        printf("\tname1: %-15s name2: %-15s\n", _parse_constraints_info[i].v1_name, _parse_constraints_info[i].v2_name);
    }
    printf("\n\n");
}

} // namespace cu_match
