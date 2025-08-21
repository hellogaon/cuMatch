#ifndef GSTREAM_GRID_FORMAT_DETAIL_SHARD_TREE_H
#define GSTREAM_GRID_FORMAT_DETAIL_SHARD_TREE_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/grid_file_stream_defines.h>
#include <ash/detail/noncopyable.h>
#include <assert.h>
#include <functional>

namespace gstream {
namespace grid_format {

struct stree_node_block;
using stree_node_ptr = stree_node_block*;
using stree_const_node_ptr = stree_node_block const*;

using stree_intn_ptr   = stree_internal_node*;
using stree_swch_ptr   = stree_switch_node*; // for LGF
using stree_leaf_ptr   = stree_leaf_node*;
// using stree_parent_ptr = stree_intn_ptr; // for LGF

#pragma pack(push, 8)
struct stree_node_header: qtree_node_header<stree_intn_ptr> { // edit for LGF
};
static_assert(sizeof(stree_node_header) == 44, "internal type size is mismatched");

struct stree_leaf_node_header: qtree_node_header<stree_swch_ptr> { // for LGF
};
static_assert(sizeof(stree_leaf_node_header) == 44, "internal type size is mismatched");

struct stree_internal_node: stree_node_header, qtree_internal_node<stree_node_ptr> {
};

struct stree_switch_node : stree_node_header, qtree_switch_node<stree_leaf_ptr> { // for LGF
};

struct stree_leaf_node: stree_leaf_node_header, qtree_leaf_node_metadata { // edit for LGF
    struct extent_info_struct {
        extent_id_t extent_id;
        uint32_t    record_id;
    } ext_info;

    struct degree_info_struct {
        degree_type_code stream_unit_type;
        unsigned char stream_unit_size;
        degree_type_code indeg_compressed_type;
        degree_type_code outdeg_compressed_type;
        uint64_t indeg_stream_size;
        uint64_t outdeg_stream_size;
    } deg_info;

};
#pragma pack(pop)

inline bool operator<(stree_leaf_node const& l, stree_leaf_node const& r) {
    return l.unique_id < r.unique_id;
}

inline bool operator>(stree_leaf_node const& l, stree_leaf_node const& r) {
    return l.unique_id > r.unique_id;
}

inline bool operator==(stree_leaf_node const& l, stree_leaf_node const& r) {
    return l.unique_id == r.unique_id;
}

inline bool is_same_sleaf(stree_leaf_node const& l, stree_leaf_node const& r) {
    return l == r;
}

void print_sleaf_info(stree_leaf_node const& sleaf);

struct stree_node_block {
    union {
        stree_node_header      header;
        stree_leaf_node_header leaf_header; // for LGF
        stree_internal_node    intn;
        stree_switch_node      swch; // for LGF
        stree_leaf_node        leaf;
    };

    operator stree_node_header&() {
        return header;
    }

    operator stree_leaf_node_header&() { // for LGF
        return leaf_header;
    }

    operator stree_internal_node&() {
        assert(header.qnode_type == qtree_node_type::InternalNode);
        return intn;
    }

    operator stree_switch_node&() { // for LGF
        assert(header.qnode_type == qtree_node_type::SwitchNode);
        return swch;
    }

    operator stree_leaf_node&() {
        assert(header.qnode_type == qtree_node_type::LeafNode);
        return leaf;
    }
};

class shard_tree: ash::noncopyable {
public:
    struct config_t {
        gbid_t gbid;
        stree_node_ptr root_node;
        qtree_node_type root_type;
        uint64_t num_nodes;
        ash::closed_interval<shard_uid> shard_range;
    };

    using traversal_callback = std::function<void(stree_node_ptr)>;
    shard_tree();
    shard_tree(config_t const& cfg);
    ~shard_tree() noexcept;

    void init(config_t const& cfg);
    void post_order_dfs(traversal_callback const& visit) const;
    void post_order_dfs_leaf_only(traversal_callback const& callback) const;
    void post_order_dfs_without_leaf(traversal_callback const& callback) const; // for LGF

    qtree_node_type root_type() const;
    stree_node_ptr  root_node() const;
    uint64_t        num_nodes() const;
    gbid_t const&   gbid() const;
    bool            empty() const;
    laddr_interval  src_range() const;
    laddr_interval  dst_range() const;
    ash::closed_interval<shard_uid> const& shard_range() const;
    uint64_t num_leaf_nodes() const;

private:
    config_t _cfg;
};

inline qtree_node_type shard_tree::root_type() const {
    return _cfg.root_type;
}

inline stree_node_ptr shard_tree::root_node() const {
    return _cfg.root_node;
}

inline uint64_t shard_tree::num_nodes() const {
    return _cfg.num_nodes;
}

inline gbid_t const& shard_tree::gbid() const {
    return _cfg.gbid;
}

inline bool shard_tree::empty() const {
    return _cfg.root_node == nullptr;
}

inline laddr_interval shard_tree::src_range() const {
    return root_node()->header.src_range;
}

inline laddr_interval shard_tree::dst_range() const {
    return root_node()->header.dst_range;
}

inline ash::closed_interval<shard_uid> const& shard_tree::shard_range() const {
    return _cfg.shard_range;
}

inline uint64_t shard_tree::num_leaf_nodes() const {
    return _cfg.shard_range.width();
}

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_DETAIL_SHARD_TREE_H
