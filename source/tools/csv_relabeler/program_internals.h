#ifndef PROGRAM_INTERNALS_H
#define PROGRAM_INTERNALS_H
#include "csv_ofstream.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <random>

class csv_relabeler {
public:
    struct config_t {
        std::string input_path;
        std::string output_path;
        std::string schema_path;
        bool has_header;
    };

    bool run(config_t const& cfg);

private:
    config_t _cfg;
    uint64_t _num_total_vertices;
    uint64_t _num_total_edges;
    uint64_t _total_num_duplicates;
    std::map<std::string, uint32_t> _vl_map;
    std::vector<std::map<uint64_t, uint64_t> > _v_map_vec;
    bool _is_vertex_file(const std::string& file_name);
    bool _is_edge_file(const std::string& file_path);
    uint64_t _count_lines(const std::string& file_path);
    void _process_edge_file(const std::string& file_path, const std::string& file_name, uint32_t src_vl_id, uint32_t dest_vl_id);
    void _process_vertex_file(const std::string& file_path, const std::string& file_name, uint32_t vl_id);
    std::string _make_input_path(std::string& path, std::string& sub_path);
};

#endif // !PROGRAM_INTERNALS_H
