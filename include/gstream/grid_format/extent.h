#ifndef GSTREAM_GRID_FORMAT_EXTENT_H
#define GSTREAM_GRID_FORMAT_EXTENT_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/grid_file_stream_defines.h>
#include <string.h> // memset

namespace gstream {
namespace grid_format {
struct extent_load_info {
    extent_id_t           ext_id;
    shard_uid             suid_start;
    shard_uid             suid_last;
    grid_off              col_beg;
    grid_off              col_last;
    uint64_t              io_offset;
    uint64_t              io_size;
    unsigned              data_offset;
    unsigned              num_shards;
    uint64_t              shard_arr_buf_size;
    uint64_t              indeg_buf_size;
    uint64_t              outdeg_buf_size;
    readonly_sleaf_ptrarr sleaf_v;
    uint64_t              required_extent_buffer_size;
    uint64_t              complete_mode_offset;

    void clear() {
        memset(this, 0, sizeof(extent_load_info));
        ext_id = InvalidExtentID;
    }
};

//uint64_t extent_buffer_size(extent_load_info const& ld_info);

struct extent_t {
    extent_load_info ld_info;
    sleaf_t* const*  sleaf_v;
    void*            buffer;
    uint64_t         buffer_size;

    void clear() {
        ld_info.clear();
        buffer      = nullptr;
        buffer_size = 0;
    }
};

struct extent_config {
    extent_load_info const* ld_info;
    void*                   buffer;
    uint64_t                buffer_size;
    uint64_t                attr_buffer_size;

    void clear() {
        memset(this, 0, sizeof(extent_t));
    }
};

void init_extent(extent_t& extent, extent_config const& cfg);

uint64_t host_extent_buffer_size(grid_file_stream* gs, extent_id_t ext_id);

} // ns grid_format
} // ns gstream
#endif // GSTREAM_GRID_FORMAT_EXTENT_H
