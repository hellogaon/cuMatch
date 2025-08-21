#ifndef GSTREAM_GRID_FORMAT_DETAIL_LABELED_FLIP24_SHARD_FORMATTER_H
#define GSTREAM_GRID_FORMAT_DETAIL_LABELED_FLIP24_SHARD_FORMATTER_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/detail/shard_tree.h>
#include <gstream/flip24_array.h>
#include <ash/bitset.h>
#include <atomic>

namespace gstream {
namespace grid_format {
namespace detail {
namespace gridgen {

using pbset_t = ash::bitset<GridBlockWidth>;

class labeled_flip24_shard_formatter {
public:
    struct config_t {
        uint64_t max_num_edges;
    };

    struct converting_result {
        uint64_t num_edges;
        uint32_t num_adj_lists;
        uint32_t num_rev_adj_lists;
        uint64_t num_duplicates;
        uint64_t num_self_loops;
        laddr_t min_src_vid;
        laddr_t max_src_vid;
        laddr_t min_dst_vid;
        laddr_t max_dst_vid;
    };

    labeled_flip24_shard_formatter();
    ~labeled_flip24_shard_formatter() noexcept;
    void init(config_t const& cfg);
    void cleanup();
    converting_result convert( // returns the number of edges after deduplication
        gbid_t gbid,
        e32_t* input_edges,
        uint64_t num_input_edges,
        bool big_endian,
        uint64_t* out_vmark_src,
        uint64_t* out_vmark_dst,
        uint32_t* out_indeg_loc,
        uint32_t* out_outdeg_loc
    );
    uint64_t get_out_edges_binary_size() const;
    uint64_t get_in_edges_binary_size() const;
    void write(gbid_t gbid, void* dst) const;

    uint32_t num_cols() const {
        assert(_buffer.num_cols <= UINT32_MAX);
        return static_cast<uint32_t>(_buffer.num_cols);
    }

    uint32_t num_rows() const {
        assert(_buffer.num_rows <= UINT32_MAX);
        return static_cast<uint32_t>(_buffer.num_rows);
    }

private:
    void _init_buffer();
    void _clear_buffer();
    void _cleanup_buffer();
    void _make_out_edges_header(gbid_t gbid, flip24_shard_header* header) const;
    void _make_in_edges_header(gbid_t gbid, flip24_shard_header* header) const;
    uint64_t _actual_size_rows() const;
    uint64_t _actual_size_cols() const;
    uint64_t _actual_size_ptrs() const;
    uint64_t _aligned_size_rows() const;
    uint64_t _aligned_size_cols() const;
    uint64_t _aligned_size_ptrs() const;
    uint64_t _actual_size_rev_rows() const;
    uint64_t _actual_size_rev_cols() const;
    uint64_t _actual_size_rev_ptrs() const;
    uint64_t _aligned_size_rev_rows() const;
    uint64_t _aligned_size_rev_cols() const;
    uint64_t _aligned_size_rev_ptrs() const;

    static constexpr unsigned NumExistedBitElements = GridBlockWidth >> 6;
    struct {
        flip24_element* rows;
        flip24_element* cols;
        colptr_t* ptrs;
        flip24_element* rev_rows;
        flip24_element* rev_cols;
        colptr_t* rev_ptrs;
        e32_t*        unique_edges;
        std::atomic<uint32_t>* local_indeg_arr;
        std::atomic<uint32_t>* local_outdeg_arr;
        std::atomic<uint64_t>* local_dst_existence_bitmap;
        colptr_t num_cols;
        uint32_t num_rows;
        uint32_t num_rev_rows;
        uint32_t num_ptrs;
        uint32_t num_rev_ptrs;
        uint32_t min_src_vid;
        uint32_t max_src_vid;
        uint32_t min_dst_vid;
        uint32_t max_dst_vid;
    } _buffer;
    config_t _cfg;
};

} // namespace gridgen
} // namespace detail
} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_DETAIL_LABELED_FLIP24_SHARD_FORMATTER_H
