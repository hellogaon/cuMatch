#include "program_internals.h"
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/sort/sort.hpp>
#include <ash/utility/prompt.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <regex>

namespace fs = ::boost::filesystem;

bool csv_relabeler::_is_vertex_file(const std::string& file_name) {
    static const std::regex vertex_pattern(R"(^\s+\.csv$)");  
    return std::regex_match(file_name, vertex_pattern);
}

bool csv_relabeler::_is_edge_file(const std::string& file_name) {
    static const std::regex edge_pattern(R"(^\s+_\s+_\s+\.csv$)");  
    return std::regex_match(file_name, edge_pattern);
}

uint64_t csv_relabeler::_count_lines(const std::string& file_path) {
    std::ifstream ifs(file_path);
    assert(ifs.is_open() && "File open error.");
    std::string filename = fs::path(file_path).stem().string();
    size_t num_items = std::count(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>(), '\n');
    if (_cfg.has_header) num_items--;
    ifs.close();
    return num_items;
}

std::string csv_relabeler::_make_input_path(std::string& path, std::string& sub_path) {
    std::string ret = path;
    ret += "/";
    ret += sub_path;
    assert(ret.length() <= 4096 && "path is too long!!!");
    return ret;
}

struct e64_t {
    uint64_t u;
    uint64_t v;
};

inline bool e64_compare(e64_t const& l, e64_t const& r) {
    if (l.u != r.u)
        return l.u < r.u;
    return l.v < r.v;
}

inline bool e64_equal(e64_t const& l, e64_t const& r) {
    return l.u == r.u && l.v == r.v;
}

void csv_relabeler::_process_edge_file(const std::string& file_path, const std::string& file_name, uint32_t src_vl_id, uint32_t dest_vl_id) {
    std::cout << "\tProcessing Edge File: " << file_path << ", src_vl_id: " << src_vl_id << ", dest_vl_id: " << dest_vl_id << std::endl;

    std::ifstream ifs(file_path);
    std::string line;

    csv_ofstream ofs;
    ofs.init_stream(_cfg.output_path + "/" + file_name);
    ofs.write_header(2);

    if (_cfg.has_header) std::getline(ifs, line);
    
    uint64_t num_edges = 0;
    while (std::getline(ifs, line)) {
        boost::tokenizer<> tok(line);
        assert(std::distance(tok.begin(), tok.end()) == 2);
        auto tok_iter = tok.begin();
        uint64_t u = _v_map_vec[src_vl_id][stoull(*tok_iter++)];
        uint64_t v = _v_map_vec[dest_vl_id][stoull(*tok_iter)];

        ofs.write_double_integer(u, v);
        num_edges++;
    }
    // assert(line_count == num_edges);

    ofs.close_stream();
    ifs.close();
}

void csv_relabeler::_process_vertex_file(const std::string& file_path, const std::string& file_name, uint32_t vl_id) {
    std::cout << "\tProcessing Vertex File: " << file_path << std::endl;

    uint64_t vertex_id = 0;
    std::ifstream ifs(file_path);
    std::string line;

    csv_ofstream ofs;
    ofs.init_stream(_cfg.output_path + "/" + file_name);
    ofs.write_header(1);

    if (_cfg.has_header) std::getline(ifs, line);

    while (std::getline(ifs, line)) {
        uint64_t v = stoull(line);
        _v_map_vec[vl_id][v] = vertex_id;

        ofs.write_single_integer(vertex_id);
        vertex_id++;
    }

    ofs.close_stream();
    ifs.close();
}

bool csv_relabeler::run(config_t const& cfg) {
    _cfg = cfg;
    _num_total_vertices = 0;
    _num_total_edges = 0;
    _total_num_duplicates = 0;

    if (!fs::is_empty(_cfg.output_path)) {
        ash::yesno_prompt prompt;
        prompt.text  = "CSV file is already exists. Do you want replace it?";
        if (!prompt()) {
            printf("CSV convertion aborted by user.\n");
            return false;
        }
        fs::remove_all(_cfg.output_path);
        fs::create_directory(_cfg.output_path);
    }

    using json = nlohmann::json;

    std::ifstream ifs(_cfg.schema_path);
    if (!ifs.is_open()) {
        printf("Schema file open error.\n");
        return false;
    }

    json full_data = json::parse(ifs);    
    {   
        uint32_t vertex_label_id = 0;
        json data = full_data["vertices"];
        _v_map_vec.resize(data.size());
        for (json::iterator it = data.begin(); it != data.end(); it++) {
            std::string vl = it.key();
            json value = it.value();

            if (value.is_string()) {
                std::string path = it.value();
                _vl_map[vl] = vertex_label_id;
                std::string full_path = _make_input_path(_cfg.input_path, path);
                _process_vertex_file(full_path, path, vertex_label_id);
                vertex_label_id++;
                if (vertex_label_id + 1 > _v_map_vec.size()) _v_map_vec.resize(vertex_label_id + 1);
            }
            else { // for multiple vertex label
                json recursive_data = value;
                for (json::iterator it2 = recursive_data.begin(); it2 != recursive_data.end(); it2++) {
                    std::string vl = it2.key();
                    json value = it2.value();

                    if (value.is_string()) {
                        std::string path = it2.value();
                        _vl_map[vl] = vertex_label_id;
                        std::string full_path = _make_input_path(_cfg.input_path, path);
                        _process_vertex_file(full_path, path, vertex_label_id);
                        vertex_label_id++;
                        if (vertex_label_id + 1 > _v_map_vec.size()) _v_map_vec.resize(vertex_label_id + 1);
                    }
                }
            }
        }
        
    }
    {
        json data = full_data["edges"];

        for (json::iterator it = data.begin(); it != data.end(); it++) {
            std::string el = it.key();
            json value = it.value();

            std::vector<std::string> paths;
            if (value.is_string()) {
                std::string path = it.value();
                paths.push_back(path); 
            }
            else { // for multiple edge label
                for (json::iterator it2 = value.begin(); it2 != value.end(); it2++) {
                    std::string path = it2.value();
                    paths.push_back(path);
                }
            }
            
            for(std::string path: paths) {
                boost::tokenizer<> tok(path);
                std::vector<std::string> token_list;
                for(boost::tokenizer<>::iterator beg = tok.begin(); beg != tok.end(); beg++) {
                    token_list.push_back(*beg);
                }
                assert(token_list.size() == 3 && "File format mismatched.");

                std::string& src_vl = token_list[0];
                std::string& dst_vl = token_list[2];

                std::string full_path = _make_input_path(_cfg.input_path, path);
                std::ifstream ifs(full_path.c_str());
                assert(ifs.is_open() && "Edge label file open error.");
                std::string filename = fs::path(full_path).stem().string();

                _process_edge_file(full_path, path, _vl_map[src_vl], _vl_map[dst_vl]);
            }
        }
    }

    return true;
}
