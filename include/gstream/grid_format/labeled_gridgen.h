#ifndef GSTREAM_GRID_FORMAT_LABELED_GRIDGEN_H
#define GSTREAM_GRID_FORMAT_LABELED_GRIDGEN_H
#include <gstream/grid_format/detail/el32_ofstream.h>
#include <gstream/grid_format/detail/input_tree.h>
#include <gstream/grid_format/detail/disk_index_tree.h>
#include <gstream/grid_format/detail/degree_block_builder.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <set>
#include <map>

namespace gstream {
namespace grid_format {
namespace detail {
namespace gridgen {

class e32_sharder;
class labeled_flip24_shard_formatter;
class dense_block_formatter;

class labeled_grid_generator {
    struct gen_state_t {
        grid_dim dim;
        uint64_t  vertex_range;
        uint64_t* vertex_mark_arr;
        uint64_t  grid_size;
        uint64_t  num_edges;
        uint64_t  num_shards;
        uint64_t  num_xtree_nodes;
        uint64_t  indeg_size;
        uint64_t  outdeg_size;
        unsigned  max_indeg_unit_size;
        unsigned  max_outdeg_unit_size;
    };

    struct io_buffer {
        void* in_buffer;
        void* out_sparse_buffer;
        void* out_dense_buffer;
        uint64_t in_buffer_size;
        uint64_t out_sparse_buffer_size;
        uint64_t out_dense_buffer_size;
        xtree_physical_pointer* xtree_pptr_arr;
        uint32_t* indeg_buffer;
        uint32_t* outdeg_buffer;
    };

    struct output_streams {
        std::ofstream* grid_stream;
        std::ofstream* xtree_stream;
    };

public:
    struct config_t {
        char const* grid_name;
        char const* indir;
        char const* outdir;
        char const* schema_path;
        uint64_t base_shard_size;
        uint64_t _max_input_shard_size;
        bool has_header;
        bool is_dense_format;
        bool skip_relabeling;
    };
    
    labeled_grid_generator();
    ~labeled_grid_generator() noexcept;
    void exec(config_t const& cfg); 

    uint32_t const el32_buffer_size = ash::MiB(1);

private:
    struct _degfile_result {
        uint64_t indeg_size;
        uint64_t outdeg_size;
        unsigned max_indeg_unit_size;
        unsigned max_outdeg_unit_size;
    };

    config_t                                     _cfg;
    gen_state_t                                  _gen_state;
    io_buffer                                    _iobuf;
    output_streams                               _ostream;
    e32_sharder*                                 _sharder;
    labeled_flip24_shard_formatter*              _sparse_fmtter;
    dense_block_formatter*                       _dense_fmtter;
    degree_block_builder*                        _deg_builder;
    std::vector<std::string>                     _input_file_list;

    // for LGF
    struct _VertexFileInfo {
        std::string full_path;
        uint64_t num_vertices;
        grid_off vertex_label_id;
        grid_off min_grid_row;
        grid_off max_grid_row;
    };

    struct _EdgeFileInfo {
        std::string full_path;
        uint64_t num_edges;
        grid_off edge_label_id;
        grid_off src_vertex_label_id;
        grid_off dst_vertex_label_id;
        grid_off num_grids_per_edge_label;
        bool is_undirected;
    };

    el32_ofstream*                               _el_ofs;
    grid_off                                     _num_vertex_labels;
    grid_off                                     _num_edge_labels;
    uint64_t                                     _num_total_vertices;
    uint64_t                                     _num_total_edges;
    uint32_t                                     _max_num_dense_row;
    grid_off                                     _num_total_row_grids;
    grid_off                                     _max_num_grids_per_edge_label;
    std::vector<_VertexFileInfo>                 _vertex_files_info;
    std::vector<_EdgeFileInfo>                   _edge_files_info;
    std::map<std::string, grid_off>              _vl_map; // vertex label to vertex label ID mapper
    std::map<std::string, grid_off>              _el_map; // edge label to edge label ID mapper
    std::vector<std::string>                     _vl_vec; // vertex label ID to vertex label mapper
    std::vector<std::string>                     _el_vec; //edge label ID to edge label mapper
    std::set<std::string>                        _undirected_el;



    bool                            _init(config_t const& cfg);
    void                            _cleanup();
    void                            _iterate_files();
    disk_index_tree                 _itree_traversal(input_tree* itree);
    void                            _make_two_leaf_nodes_for_switch_node(gbid_t const& gbid, input_tree::leaf_node_t const* ileaf, xtree_internal_node* xnode_parent, char shard_row, char shard_col, xtree_switch_node& xswch);
    void                            _serialize_xtree(disk_index_tree* xtree);
    void                            _serialize_grid_info() const;
    void                            _serialize_label() const;
    _degfile_result                 _serialize_degree() const;

    // for LGF
    void                            _get_vertex_files_info(grid_off& vertex_label_id, nlohmann::json& data);
    void                            _get_edge_files_info(grid_off& edge_label_id, nlohmann::json& data);
    bool                            _parse_json();
    void                            _generate_grid();
};

} // namespace gridgen
} // namespace detail
} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_LABELED_GRIDGEN_H
