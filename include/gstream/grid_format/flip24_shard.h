#ifndef GSTREAM_GRID_FORMAT_FLIP24_SHARD
#define GSTREAM_GRID_FORMAT_FLIP24_SHARD
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/flip24_array.h>
#include <gstream/internal_defines.h>
#include <ash/numeric.h>
#include <assert.h>

namespace gstream {
namespace grid_format {

template <typename T>
GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
static constexpr T _cuda_aligned_size(T const size) {
    return ash::aligned_size(size, InternalDataSectionAlignment);
}

struct flip24_shard_header {
    gbid_t gbid;
    struct {
        /* Matrix block represents edges by 24-bit local address.
         * Since the GPU is sensitive to data alignment,
         * we partition the 24-bit address to high 8-bit and low 16-bit and store to separated vectors.
         * Each edge representation requires two address pairs: {row, column}, so the total number of vectors is four.
         */
        uint32_t rowv; // offset of 24-bit row vector
        uint32_t colv; // offset of 24-bit column vector
    } section_offset;
    uint32_t min_src_vid; // for LGF
    uint32_t max_src_vid; // for LGF
    uint32_t num_rows; // for LGF
    uint32_t num_cols;
};

static_assert(sizeof(flip24_shard_header) == 36, "The size of matrix block header is not 36"); // for LGF

class flip24_shard {
public:
    struct adj_list_t {
        laddr_t src_vertex;
        uint32_t length;
        flip24_element const* _colv;
        uint32_t _colptr;

        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        laddr_t operator[](uint32_t const i) const {
            assert(i < length);
            assert(_is_aligned_address(_colv, 4));
            return read_flip24_element(_colv, static_cast<uint64_t>(i) + _colptr);
        }
    };

    // Get the number of adjacency lists in the block
    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t num_lists() const {
        // (PTRVEC size - Dummy pointer size) / 4
        return header.num_rows; // for LGF
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t num_edges() const {
        return header.num_cols;
        //return n * 0x55555556 >> 1; // n / 3

       /* 64-bit */
       //return n * 12297829382473034411ULL; // n / 3
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t min_src_vid() const { // for LGF
        return header.min_src_vid;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t max_src_vid() const { // for LGF
        return header.max_src_vid;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t len_src_vid() const { // for LGF
        return header.max_src_vid - header.min_src_vid + 1;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint64_t block_size() const {
        return _aligned_size(sizeof(flip24_element) * header.num_cols, InternalDataSectionAlignment) + header.section_offset.colv;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    gbid_t gbid() const {
        return header.gbid;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    adj_list_t adj_list(laddr_t const i) const { // for LGF, for src_v idx
        adj_list_t list;
        list.src_vertex = source_vertex(i);
        uint32_t ptr_idx = list.src_vertex - header.min_src_vid;
        list.length = _list_size(ptr_idx);
        list._colv = colv();
        list._colptr = colptr(ptr_idx);
        return list;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    adj_list_t adj_list_vid(laddr_t const v_id) const { // for LGF, for vid
        adj_list_t list;
        list.src_vertex = v_id;
        if (header.min_src_vid <= v_id && v_id <= header.max_src_vid) {
            uint32_t ptr_idx = v_id - header.min_src_vid;
            list.length = _list_size(ptr_idx);
            list._colv = colv();
            list._colptr = colptr(ptr_idx);
            return list;
        }
        else {
            list.length = 0;
            list._colv = colv();
            list._colptr = 1; // for optional
            return list;
        }
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    bool valid_source_vertex(laddr_t const v_id) const { // for LGF, for vid
        if (header.min_src_vid <= v_id && v_id <= header.max_src_vid) {
            uint32_t ptr_idx = v_id - header.min_src_vid;
            return _list_size(ptr_idx) > 0;
        }
        else return false;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    laddr_t source_vertex(laddr_t const i) const {
        //laddr_t const src = read_flip24v_element(_seek_pointer(this, header.section_offset.rowv), i);
        laddr_t const src = read_flip24_element(rowv(), i);
        return src;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    uint32_t _list_size(laddr_t const i) const {
        return colptr(i + 1) - colptr(i);
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    void const* _data() const {
        return &header + 1;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    flip24_element const* rowv() const {
        flip24_element const* v = reinterpret_cast<flip24_element const*>(_seek_pointer(this, header.section_offset.rowv));
        return v;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        flip24_element const* colv() const {
        flip24_element const* v = reinterpret_cast<flip24_element const*>(_seek_pointer(this, header.section_offset.colv));
        return v;
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
    colptr_t colptr(laddr_t const row_off = 0) const {
        colptr_t const* v = static_cast<colptr_t const*>(_data());
        return v[row_off];
    }

    flip24_shard_header header;

private:
    GSTREAM_DEVICE_COMPATIBLE static bool _is_aligned_address(void const* addr, uint64_t const alignment) {
    static_assert(sizeof(void*) <= sizeof(uint64_t), "Pointer size is greater than 8!");
        auto const addr2 = reinterpret_cast<uint64_t>(addr);
        return addr2 % alignment == 0;
    }

    template <typename T1, typename T2>
    GSTREAM_DEVICE_COMPATIBLE
    static constexpr T1 _aligned_size(const T1 size, const T2 align) {
        return align * ((size + align - 1) / align);
    }

    template <typename T>
    GSTREAM_DEVICE_COMPATIBLE
    static T* _seek_pointer(T* p, int64_t offset) {
        static_assert(sizeof(void*) <= sizeof(uint64_t), "Pointer size is greater than 8!");
        auto const p2 = (char*)p;
        return reinterpret_cast<T*>(p2 + offset);
    }
};

GSTREAM_HOST_ONLY void print_shard_info(flip24_shard const* shard);

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_FLIP24_SHARD
