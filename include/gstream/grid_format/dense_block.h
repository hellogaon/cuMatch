#ifndef GSTREAM_GRID_FORMAT_DENSE_BLOCK_H
#define GSTREAM_GRID_FORMAT_DENSE_BLOCK_H
#include <gstream/grid_format/detail/shard_tree.h>
#include <assert.h>

namespace gstream {
namespace grid_format {

struct dense_format_compressed_info_t {
    uint8_t rowv_diff;
    uint8_t adj_sz;
    uint8_t col_hi;
    uint8_t col_lo;
};

struct dense_block_header {
    dense_format_compressed_info_t compressed_info;
    struct {
        uint32_t adj_sz;
        uint32_t col_hi;
        uint32_t col_lo;
    } section_offset;
    uint32_t num_rows;
    uint32_t num_cols;
};

static_assert(sizeof(dense_block_header) == 24, "The size of dense block header is not 16");

class dense_block {
public:
    struct list_t {
        uint8_t bit_sz;
        uint32_t length;
        uint8_t const* arr;
        uint32_t operator[](uint32_t const i) const {
            assert(i < length);
            return _read_arbitrary_bit(arr, bit_sz, i);
        }
    };

    struct dense_block_data {
        list_t rowv_diff;
        list_t adj_sz;
        list_t col_hi;
        list_t col_lo;
    };

    dense_block_data data() const {
        return {
            {header.compressed_info.rowv_diff, header.num_rows, rowv_diff()},
            {header.compressed_info.adj_sz, header.num_rows, adj_sz()},
            {header.compressed_info.col_hi, header.num_rows, col_hi()},
            {header.compressed_info.col_lo, header.num_cols, col_lo()}
        };
    }

    ASH_FORCEINLINE
    uint32_t get_num_rows() const {
       return header.num_rows;
    }

    ASH_FORCEINLINE
    uint32_t get_num_cols() const {
       return header.num_cols;
    }

    ASH_FORCEINLINE
    uint8_t get_col_lo_bit() const {
       return header.compressed_info.col_lo;
    }

    ASH_FORCEINLINE
    uint8_t get_col_high_bit() const {
        return header.compressed_info.col_hi;
    }

    ASH_FORCEINLINE
    dense_format_compressed_info_t get_compressed_info() const {
        return header.compressed_info;
    }

    void dense_to_sparse(sleaf_t const& sleaf, void* dst, void* dense_ptr) const;
    void dense_to_sparse_with_parallel(sleaf_t const& sleaf, void* dst, void* dense_ptr) const;
    void dense_to_sparse_without_parallel(sleaf_t const& sleaf, void* dst) const;
    dense_block_header header;

private:
    ASH_FORCEINLINE
    void const* _data() const {
        return &header + 1;
    }

    ASH_FORCEINLINE
    uint8_t const* rowv_diff() const { 
        uint8_t const* v = reinterpret_cast<uint8_t const*>(_data());
        return v;
    }

    ASH_FORCEINLINE
    uint8_t const* adj_sz() const { 
        uint8_t const* v = reinterpret_cast<uint8_t const*>((_seek_pointer(this, header.section_offset.adj_sz)));
        return v;
    }

    ASH_FORCEINLINE
    uint8_t const* col_hi() const {
        uint8_t const* v = reinterpret_cast<uint8_t const*>(_seek_pointer(this, header.section_offset.col_hi));
        return v;
    }

    ASH_FORCEINLINE
    uint8_t const* col_lo() const {
        uint8_t const* v = reinterpret_cast<uint8_t const*>(_seek_pointer(this, header.section_offset.col_lo));
        return v;
    }

    ASH_FORCEINLINE
    uint64_t _actual_size_rows(sleaf_t const& sleaf) const {
        return sizeof(flip24_element) * sleaf.num_adj_lists;
    }

    ASH_FORCEINLINE
    uint64_t _actual_size_cols(sleaf_t const& sleaf) const {
        return sizeof(flip24_element) * sleaf.num_edges;
    }

    ASH_FORCEINLINE
    uint64_t _actual_size_ptrs(sleaf_t const& sleaf) const {
        return sizeof(colptr_t) * (sleaf.max_src_vid - sleaf.min_src_vid + 2); // for LGF
    }

    ASH_FORCEINLINE
    uint64_t _aligned_size_rows(sleaf_t const& sleaf) const {
        return ash::aligned_size(_actual_size_rows(sleaf), InternalDataSectionAlignment);
    }

    ASH_FORCEINLINE
    uint64_t _aligned_size_cols(sleaf_t const& sleaf) const {
        return ash::aligned_size(_actual_size_cols(sleaf), InternalDataSectionAlignment);
    }

    ASH_FORCEINLINE
    uint64_t _aligned_size_ptrs(sleaf_t const& sleaf) const {
        return ash::aligned_size(_actual_size_ptrs(sleaf), InternalDataSectionAlignment);
    }

    void _make_sparse_header(sleaf_t const& sleaf, flip24_shard_header* header_out) const;
    void _clear_buffer(sleaf_t const& sleaf, void* dst) const;

    static uint32_t _read_arbitrary_bit(uint8_t const* b, uint8_t const& bit_sz, uint32_t const& index) {
        uint32_t      ret     = 0;
        uint32_t      b_index = static_cast<uint64_t>(bit_sz) * index / 8;
        uint8_t const b_left  = 8 - ((bit_sz * index) % 8);
        if (b_left >= bit_sz) {
            uint32_t mask = (1U << bit_sz) - 1;
            ret = (b[b_index] >> (b_left - bit_sz)) & mask;
        }
        else {
            uint8_t const  left = bit_sz - b_left;
            uint32_t const mask = (1U << b_left) - 1;
            ret |= ((b[b_index] & mask) << left);
            for (int8_t i = left; i > 0; i -= 8) {
                if (i < 8) ret |= (b[++b_index] >> (8 - i));
                else ret |= (b[++b_index] << (i - 8));
            }
        }

        // Version 1
        // uint32_t ret = 0;
        // uint32_t b_index = (uint64_t)bit_sz * index / 8;
        // uint32_t start = (bit_sz * index) % 8;
        // for(int i = 0; i < bit_sz; i++) {gb
        //     uint32_t new_n_index = b_index + (start + i) / 8;
        //     ret <<= 1;
        //     ret |= (b[new_n_index] >> ((7 - ((start + i) % 8))) & 1);
        // }

        return ret;
    }
    
    template <typename T>
    static T* _seek_pointer(T* p, int64_t offset) {
        static_assert(sizeof(void*) <= sizeof(uint64_t), "Pointer size is greater than 8!");
        auto const p2 = (char*)p;
        return reinterpret_cast<T*>(p2 + offset);
    }
};

void convert_dense_to_sparse(dense_block const* src, flip24_shard* dst, sleaf_t* sleaf, void* dense_ptr);
void convert_dense_to_sparse_with_parallel(dense_block const* src, flip24_shard* dst, sleaf_t* sleaf, void* dense_ptr);
void convert_dense_to_sparse_without_parallel(dense_block const* src, flip24_shard* dst, sleaf_t* sleaf);

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_DENSE_BLOCK_H