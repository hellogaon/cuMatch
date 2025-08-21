// Change for LGF (Labeled)
#ifndef GSTREAM_GRID_FORMAT_DEFINES_H
#define GSTREAM_GRID_FORMAT_DEFINES_H
#include <ash/numeric.h>
#include <gstream/internal_defines.h>
#include <gstream/flip24_array.h>
#include <ash/vectorization.h>
#include <ash/pointer.h>
#include <utility>
#include <limits>

namespace gstream {
namespace grid_format {

using laddr_hi = uint8_t; // High 8-bits of local address  [seg_off_t -> seg_id_t (1/24/2020) -> laddr_hi (3/2/2020)]
constexpr laddr_hi MaxSegmentID = std::numeric_limits<laddr_hi>::max();
using laddr_lo = uint16_t; // Low 16-bits of local address
using colptr_t = uint32_t;
using grid_dim = ash::dim3<grid_off>;

constexpr unsigned LocalAddrBits = 24;
constexpr unsigned LocalAddrLowBits = 16;
constexpr unsigned LocalAddrHighBits = 8;
constexpr unsigned GridBlockWidth = 1u << LocalAddrBits;
constexpr uint64_t MaxEdgesPerShard = static_cast<uint64_t>(GridBlockWidth) * static_cast<uint64_t>(GridBlockWidth);
constexpr unsigned LabelSizeLimit = 32;

constexpr grid_off MinimumGridOffset = 0;
constexpr grid_off MaximumGridOffset = std::numeric_limits<grid_off>::max() - 1;
constexpr grid_off InvalidGridOffset = std::numeric_limits<grid_off>::max();

using shard_uid = uint64_t; // Shard Unique Identifer
constexpr shard_uid InvalidShardUnqiueID = std::numeric_limits<shard_uid>::max();

// class  disk_index_tree;
class  labeled_disk_index_tree; // for LGF
struct xtree_disk_node;
struct xtree_switch_node; // for LGF
struct xtree_internal_node;
struct xtree_leaf_node;

// using xtree_t = disk_index_tree;
using xtree_t = labeled_disk_index_tree; // for LGF
using xintn_t = xtree_internal_node;
using xswch_t = xtree_switch_node;
using xleaf_t = xtree_leaf_node;

typedef struct gbid_struct {
    struct compressed_t {
        uint64_t grid_id;
        uint32_t label_id;

        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator>(compressed_t const& rhs) const {
            if (grid_id == rhs.grid_id) return label_id > rhs.label_id;
            return grid_id > rhs.grid_id;
        }
        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator<(compressed_t const& rhs) const {
            if (grid_id == rhs.grid_id) return label_id < rhs.label_id;
            return grid_id < rhs.grid_id;
        }
        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator>=(compressed_t const& rhs) const {
            if (grid_id == rhs.grid_id) return label_id >= rhs.label_id;
            return grid_id >= rhs.grid_id;
        }
        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator<=(compressed_t const& rhs) const {
            if (grid_id == rhs.grid_id) return label_id <= rhs.label_id;
            return grid_id <= rhs.grid_id;
        }
        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator==(compressed_t const& rhs) const {
            if (grid_id == rhs.grid_id) return label_id == rhs.label_id;
            return false;
        }
        GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator!=(compressed_t const& rhs) const {
            if (grid_id == rhs.grid_id) return label_id != rhs.label_id;
            return true;
        }
    };

    grid_off row;
    grid_off col;
    grid_off label; // edge label

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        compressed_t compressed() const {
        return *reinterpret_cast<compressed_t const*>(this);
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        operator compressed_t() const {
        return compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator>(gbid_struct const& rhs) const {
        return compressed() > rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator<=(gbid_struct const& rhs) const {
        return compressed() <= rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator>=(gbid_struct const& rhs) const {
        return compressed() >= rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator==(gbid_struct const& rhs) const {
        return compressed() == rhs.compressed();
    }

    GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
        bool operator!=(gbid_struct const& rhs) const {
        return compressed() != rhs.compressed();
    }

    ASH_FORCEINLINE void set(grid_off const row_, grid_off const col_) {
        row = row_;
        col = col_;
        label = InvalidGridOffset;
    }

    ASH_FORCEINLINE void set(grid_off const row_, grid_off const col_, grid_off const label_) {
        row = row_;
        col = col_;
        label = label_;
    }

    ASH_FORCEINLINE void set(compressed_t const compressed) {
        *reinterpret_cast<compressed_t*>(this) = compressed;
    }

} gbid_t;
static_assert(sizeof(gbid_t) == 12, "a size of gbid_struct is not 12");

ASH_FORCEINLINE gbid_t make_gbid(grid_off const row, grid_off const col) {
    return gbid_t{ row, col };
}

ASH_FORCEINLINE gbid_t make_gbid(grid_off const row, grid_off const col, grid_off const label) { // for LGF
    return gbid_t{ row, col, label };
}

ASH_FORCEINLINE uint64_t columnar_offset( // for LGF
    grid_off const row, grid_off const col, grid_off const label, grid_dim const& dim) {
    uint64_t r = col;
    r *= dim.y;
    r += row;
    r *= dim.z;
    r += label;
    return r;
}

ASH_FORCEINLINE uint64_t columnar_offset(gbid_t const& gbid, grid_dim const& dim) {  // for LGF
    return columnar_offset(gbid.row, gbid.col, gbid.label, dim);
}

enum class grid_block_type {
    Segmented,
    Matrix,
};

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
grid_off get_grid_offset(gaddr_t gaddr) {
    return static_cast<grid_off>(gaddr >> LocalAddrBits);
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
gaddr_t laddr_to_gaddr(laddr_t const local, grid_off const off) {
    gaddr_t g = off;
    g <<= 24;
    g |= local;
    return g;
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
laddr_t gaddr_to_laddr(gaddr_t gaddr) {
#ifndef __CUDACC__
    static_assert(ash::is_power_of_two(GridBlockWidth), "GridBlockWidth is must be power of two to use bitwise converting logic");
#endif
    return static_cast<laddr_t>(gaddr & (GridBlockWidth - 1));
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
laddr_t make_local_addr(laddr_lo const lo16, laddr_hi const hi8) {
    laddr_t l = hi8;
    l <<= 16;
    l |= lo16;
    return l;
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
laddr_lo get_laddr_lo(laddr_t const laddr) {
#ifndef __CUDACC__
    static_assert(ash::is_power_of_two(GridBlockWidth), "GridBlockWidth is must be power of two to use bitwise converting logic");
#endif
    constexpr laddr_t enable_bits = UINT16_MAX;
    return static_cast<laddr_lo>(laddr & enable_bits);
}

GSTREAM_DEVICE_COMPATIBLE ASH_FORCEINLINE
laddr_hi get_laddr_hi(laddr_t const laddr) {
#ifndef __CUDACC__
    static_assert(ash::is_power_of_two(GridBlockWidth), "GridBlockWidth is must be power of two to use bitwise converting logic");
#endif
    return static_cast<laddr_hi>(laddr >> LocalAddrLowBits);
}

struct global_edge_t {
    gaddr_t src;
    gaddr_t dst;
};

struct e32_t {
    laddr_t u;
    laddr_t v;
};

inline bool e32_compare(e32_t const& l, e32_t const& r) {
    if (l.u != r.u)
        return l.u < r.u;
    return l.v < r.v;
}

inline bool e32_compare_rev(e32_t const& l, e32_t const& r) { // for LGF
    if (l.v != r.v)
        return l.v < r.v;
    return l.u < r.u;
}

inline bool e32_equal(e32_t const& l, e32_t const& r) {
    return l.u == r.u && l.v == r.v;
}

inline bool e32_is_diagonal(e32_t const& e) {
    return e.u == e.v;
}

inline bool e32_is_not_diagonal(e32_t const& e) {
    return e.u != e.v;
}

struct flip24_shard_header;
class flip24_shard;

using f24_shard_ptr = flip24_shard*;
using f24_shard_cptr = flip24_shard const*;
using readonly_f24_shard_ptrarr = f24_shard_cptr const*;

template <typename T = void*>
struct unified_pointer {
    static_assert(std::is_pointer_v<T>, "A template argument T is must be pointer type");
    union {
        uint64_t disk_data_offset;
        T in_memory_addr;
    };
};

enum class qtree_node_type : char {
    NullNode = 0,
    InternalNode = 1,
    SwitchNode = 2, // for LGF
    LeafNode = 3
};

//! [Mark: qtree-traversal-order]
// NW->SW->NE->SE
enum class qtree_location : char {
    NW = 0,
    SW = 1,
    NE = 2,
    SE = 3
};

constexpr uint16_t MaxShardLevel = 8;

//! [Mark: qtree-traversal-order]
inline qtree_location make_location_id(char const row, char const col) {
    int i = col << 1;
    i |= row;
    return static_cast<qtree_location>(i);
}

//! [Mark: qtree-traversal-order]
inline char const* location_to_string(qtree_location const location) {
    static char const* s[] = { "NW", "SW", "NE", "SE" };
    return s[static_cast<int>(location)];
}

char constexpr GridFileExt[]  = ".grid";
char constexpr XtreeFileExt[] = ".xtree";
char constexpr InfoFileExt[]  = ".grid_info";
char constexpr LabelInfoFileExt[]  = ".label_info"; // for LGF
char constexpr InDegFileExt[]  = ".indeg";
char constexpr OutDegFileExt[]  = ".outdeg";
char constexpr DegIdxFileExt[] = ".degidx";

#define GRID_OPT_DISK_FORMAT_COMPRESSION 1
using grid_opt_t = unsigned;

struct grid_storage_info {
    char name[128];
    grid_dim dim;
    uint64_t vertex_range;
    uint64_t num_ext_vertices;
    uint64_t grid_size;
    uint64_t num_edges;
    uint64_t xtree_size;
    uint64_t num_shards;
    uint64_t indeg_size;
    uint64_t outdeg_size;
    unsigned max_indeg_unit_size;
    unsigned max_outdeg_unit_size;
    uint64_t base_shard_size;
    grid_opt_t grid_option;
    uint32_t num_vertex_labels; // for LGF
    uint32_t num_edge_labels; // for LGF
    uint32_t max_num_dense_row; // for LGF
};

struct vertex_label_info { // for LGF
    grid_off min_grid_row;
    grid_off max_grid_row;
    char name[LabelSizeLimit];
};
struct edge_label_info { // for LGF
    char name[LabelSizeLimit];
};

inline uint64_t degree_stream_buffer_size(unsigned const num_rows, unsigned char stream_unit_size) {
    uint64_t size = num_rows;
    size *= stream_unit_size;
    return ash::aligned_size(size, DegreeAlignment);
}

template <typename T>
struct degree_block {
    using value_type = T;
    static constexpr value_type limit = std::numeric_limits<value_type>::max();

    T data[GridBlockWidth];

    value_type operator[](laddr_t const off) const {
        return get(off);
    }
    value_type get(laddr_t off) const {
        return data[off];
    }
    void set(laddr_t off, value_type val) {
        data[off] = val;
    }
};

template <>
struct degree_block<flip24_element>{
    using value_type = flip24_element;
    static constexpr uint32_t limit = 1u << 24;

    flip24_element data[GridBlockWidth];

    uint32_t operator[](laddr_t off) const {
        return get(off);
    }
    uint32_t get(laddr_t const off) const {
        assert(ash::is_aligned_address(data, 8));
        return read_flip24_element(data, off);
    }
    void set(laddr_t const off, uint32_t const u32) {
        assert(ash::is_aligned_address(data, 8));
        assert(u32 <= limit);
        write_flip24_element(data, off, get_laddr_lo(u32), get_laddr_hi(u32));
    }
    void set(laddr_t const off, laddr_lo const lo, laddr_hi const hi) {
        assert(ash::is_aligned_address(data, 8));
        write_flip24_element(data, off, lo, hi);
    }
};

struct degree_block_info {
    degree_type_code compressed_type;
    uint64_t offset;
};

using degree_block_u8  = degree_block<uint8_t>;
using degree_block_u16 = degree_block<uint16_t>;
using degree_block_u24 = degree_block<flip24_element>;
using degree_block_u32 = degree_block<uint32_t>;
using degree_block_u64 = degree_block<uint64_t>;

using laddr_interval = ash::closed_interval<laddr_t>;
using gaddr_interval = ash::closed_interval<gaddr_t>;

inline gaddr_interval laddr_to_gaddr(laddr_interval const& l, grid_off const off) {
    return gaddr_interval {
        laddr_to_gaddr(l.hi, off),
        laddr_to_gaddr(l.lo, off)
    };
}

using suid_interval = ash::closed_interval<shard_uid>;

enum class shard_type_code: uint16_t {
    Flip24, // Current version supports only this type
    WeightedFlip24,
};

#pragma pack(push, 1)
template <typename NodePointer>
struct qtree_node_header {
    NodePointer     parent;     // 8-byte
    gbid_t          gbid;       // 12-byte for LGF
    uint16_t        level;      // 2-byte ---+
    qtree_node_type qnode_type; // 1-byte    |--> 8-byte
    qtree_location  location;   // 1-byte    |
    uint32_t        _unused;    // paddding -+ // 4-byte
    laddr_interval  src_range;  // 8-byte
    laddr_interval  dst_range;  // 8-byte
};

template <typename NodePointer>
struct qtree_internal_node {
    //! [Mark: qtree-traversal-order
    NodePointer     nw;      // 8-byte
    NodePointer     sw;      // 8-byte
    NodePointer     ne;      // 8-byte
    NodePointer     se;      // 8-byte
    qtree_node_type nw_type; // 1-byte --+
    qtree_node_type sw_type; // 1-byte   |
    qtree_node_type ne_type; // 1-byte   |--> 8-byte
    qtree_node_type se_type; // 1-byte   |
    int32_t         _unused; // 4-byte --+

    NodePointer& child_ptr(unsigned row, unsigned col) {
        //! [Mark: qtree-traversal-order]
        assert(row < 2 && "index overflow error.");
        assert(col < 2 && "index overflow error.");
        col <<= 1;
        row |= col;
        return (&nw)[row];
    }

    qtree_node_type& child_type(unsigned row, unsigned col) {
        //! [Mark: qtree-traversal-order]
        assert(row < 2 && "index overflow error.");
        assert(col < 2 && "index overflow error.");
        col <<= 1;
        row |= col;
        return (&nw_type)[row];
    }
};

template <typename NodePointer>
struct qtree_switch_node {
    NodePointer     out_edges; // 8-byte
    NodePointer     in_edges;  // 8-byte
};

struct qtree_leaf_node_metadata {
    shard_uid unique_id;       // 8-byte
    uint64_t  physical_offset; // 8-byte
    double    density;         // 8-byte
    uint64_t  num_edges;       // 8-byte
    // ====== 32-byte

    uint32_t        num_adj_lists; // 4-byte
    laddr_t         min_src_vid;   // 4-byte
    laddr_t         max_src_vid;   // 4-byte
    laddr_t         min_dst_vid;   // 4-byte
    laddr_t         max_dst_vid;   // 4-byte
    shard_type_code type_code;     // 2-byte
    uint8_t         _unused[10];   // 10-byte
    // ====== 32-byte

    struct size_info_struct {
        uint64_t in_memory_size;
        uint64_t disk_size;
    } size_info;
    // ====== 16-byte

    // ====== Total 80-byte

    laddr_interval actual_src_range() const {
        return laddr_interval { min_src_vid, max_src_vid };
    }

    laddr_interval actual_dst_range() const {
        return laddr_interval { min_dst_vid, max_dst_vid };
    }
};
#pragma pack(pop)

} // namespace grid_format

using grid_format::gbid_t;
using grid_format::flip24_shard;
using grid_format::f24_shard_ptr;
using grid_format::f24_shard_cptr;
using grid_format::readonly_f24_shard_ptrarr;

} // namespace gstream
#endif // GSTREAM_GRID_FORMAT_DEFINES_H
