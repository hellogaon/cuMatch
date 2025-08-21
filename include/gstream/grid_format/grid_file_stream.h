#ifndef GSTREAM_GRID_FORMAT_GRID_FILE_STREAM
#define GSTREAM_GRID_FORMAT_GRID_FILE_STREAM
#include <gstream/grid_format/grid_file_stream_defines.h>
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/detail/shard_tree.h>
#include <gstream/internal_defines.h>
#include <ash/io/binary_file_stream.h>
#include <ash/detail/noncopyable.h>
#include <deque>
#include <map>
#include <string>

namespace gstream {
namespace grid_format {

using grid_stream_opt = unsigned;
constexpr static grid_stream_opt GFS_USE_DIRECT_IO = 0x01;
constexpr static grid_stream_opt GFS_USE_OUT_DEGREE = 0x01 << 1;
constexpr static grid_stream_opt GFS_USE_IN_DEGREE = 0x01 << 2;

struct grid_stream_info {
    uint64_t max_shard_size;
    uint64_t min_extent_buffer_size;
    uint64_t max_extent_buffer_size;
    uint64_t max_num_contiguous_shards;
    uint64_t num_extents;
    uint64_t complete_extent_buffer_size;
    uint64_t complete_degree_buffer_size;
};

class grid_file_stream final: ash::noncopyable {
public:
    struct config_t {
        char const*     grid_name;
        char const*     grid_dir;
        grid_stream_opt stream_opt;
        uint64_t        extent_size;
    };

    struct iterator;

    grid_file_stream() = default;
    ~grid_file_stream() noexcept;

    stree_t const* stree(gbid_t const& gbid) const {
        return &_buffer.stree[columnar_offset(gbid, _storage_info.dim)];
    }

    stree_t const* stree(grid_off const row, grid_off const col, grid_off const label) const { // for LGF
        return stree(make_gbid(row, col, label));
    }

    sleaf_t** sleaf(shard_uid const suid) const {
        assert(_buffer.suid_index_v[suid] != nullptr);
        return &_buffer.suid_index_v[suid];
    }

    config_t const& config() const {
        return _cfg;
    }

    extent_load_info const&  ext_ld_info(extent_id_t const& ext_id) const;

    degree_block_info const& indeg_info(grid_off off) const {
        return _indeg_info[off];
    }

    degree_block_info const& outdeg_info(grid_off off) const {
        return _outdeg_info[off];
    }

    grid_storage_info const& storage_info() const {
        return _storage_info;
    }

    grid_stream_info const& stream_info() const {
        return _stream_info;
    }

    bool open(config_t const& cfg);
    void close();

    iterator begin() const;
    iterator end() const;
    bool     load_outdeg(void* buf);
    bool     load_indeg(void* buf);
    bool     load_in_memory_shard(void* buf, sleaf_ptr const& sleaf) const; // for LGF
    bool     load_disk_shard(void* buf, sleaf_ptr const& sleaf) const; // for LGF
    bool     load_extent(void* buf, extent_load_info const& ld_info) const;
    bool     load_extent(void* buf, extent_id_t ext_id) const;

    uint64_t num_shards() const {
        return _storage_info.num_shards;
    }

    uint64_t max_shard_size() const {
        return _stream_info.max_shard_size;
    }

    uint64_t max_extent_size() const {
        return _stream_info.max_extent_buffer_size;
    }

    uint64_t max_contigous_shards() const {
        return _stream_info.max_num_contiguous_shards;
    }

    uint64_t num_extents() const {
        return _stream_info.num_extents;
    }

    uint64_t complete_extent_buffer_size() const {
        return _stream_info.complete_extent_buffer_size;
    }

    bool is_open() const {
        return _stream.grid != nullptr;
    }

    uint64_t num_vertex_labels() const { // for LGF
        return _storage_info.num_vertex_labels;
    }

    uint64_t num_edge_labels() const { // for LGF
        return _storage_info.num_edge_labels;
    }

    grid_off get_vertex_label_id(std::string& label) const { // for LGF
        assert(_vl_map.find(label) != _vl_map.end());
        return _vl_map.find(label)->second;
    }

    grid_off get_edge_label_id(std::string& label) const { // for LGF
        assert(_el_map.find(label) != _el_map.end());
        return _el_map.find(label)->second;
    }

    bool has_vertex_label(std::string& label) const { // for LGF
        return _vl_map.find(label) != _vl_map.end();
    }

    bool has_edge_label(std::string& label) const { // for LGF
        return _el_map.find(label) != _el_map.end();
    }

    std::string get_vertex_label(grid_off lid) const { // for LGF
        assert(lid < num_vertex_labels());
        return _vl_info[lid].name;
    }

    std::string get_edge_label(grid_off lid) const { // for LGF
        assert(lid < num_edge_labels());
        return _el_info[lid].name;
    }

    grid_off get_min_grid_row(grid_off lid) const { // for LGF
        assert(lid < num_vertex_labels());
        return _vl_info[lid].min_grid_row;
    }

    grid_off get_max_grid_row(grid_off lid) const { // for LGF
        assert(lid < num_vertex_labels());
        return _vl_info[lid].max_grid_row;
    }

private:
    void _cleanup();
    void _error_log(gstream_return_code err);
    bool _load_file_buffered(void* buf, char const* path);

    bool _load_metadata();
    bool _open_grid_stream();
    bool _mapping_extents();
    void _make_stream_info();

    config_t _cfg {};
    grid_storage_info _storage_info {};
    grid_stream_info _stream_info {};

    struct _buffer_struct {
        stree_t* stree;
        stree_node_block* stree_blocks;
        sleaf_t** suid_index_v;
        degree_block_info* degidx;
    } _buffer {};

    struct _stream_struct {
        ash::binary_file_stream* grid;
        std::ifstream* indeg;
        std::ifstream* outdeg;
    } _stream {};

    std::map<std::string, grid_off> _vl_map; // vertex label to vertex label ID mapper (for LGF)
    std::map<std::string, grid_off> _el_map; // edge label to edge label ID mapper (for LGF)

    vertex_label_info*     _vl_info {}; // for LGF
    edge_label_info*       _el_info {}; // for LGF
    degree_block_info*     _indeg_info {};
    degree_block_info*     _outdeg_info {};
    detail::extent_mapper* _ext_mapper {};
    std::deque<gstream_return_code> _err_stk {};
};

struct grid_file_stream::iterator final {
    friend class grid_file_stream;
    using end_type = std::nullptr_t;

    explicit iterator(grid_file_stream const& gs);
    iterator(grid_file_stream const& gs, end_type const&);
    iterator  operator++();
    iterator  operator++(int);
    iterator& operator=(end_type const&);
    bool operator==(iterator const& other) const;
    bool operator==(end_type const&) const;
    bool operator!=(iterator const& other) const;
    bool operator!=(end_type const&) const;
    stree_t const& operator*() const;

private:
    grid_dim _dim;
    stree_t const* _stree;
    laddr_t _row;
    laddr_t _col;
    laddr_t _label; // for LGF
};

void print_grid_info2(grid_storage_info const& info) noexcept;

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_GRID_FILE_STREAM
