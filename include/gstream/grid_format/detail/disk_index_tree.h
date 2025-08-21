#ifndef GSTREAM_GRID_FORMAT_DETAIL_DISK_INDEX_TREE_H
#define GSTREAM_GRID_FORMAT_DETAIL_DISK_INDEX_TREE_H
#include <gstream/grid_format/grid_format_defines.h>
#include <ash/detail/noncopyable.h>
#include <assert.h>
#include <functional>
#include <string.h>

namespace gstream {
namespace grid_format {

using  _xtree_node_ptr = unified_pointer<xtree_disk_node*>;
using  _xtree_internal_ptr = unified_pointer<xtree_internal_node*>;
using  _xtree_leaf_ptr = unified_pointer<xtree_leaf_node*>;

#pragma pack(push, 8)
struct xtree_node_header: qtree_node_header<_xtree_internal_ptr> {
};
static_assert(sizeof(xtree_node_header) == 44, "internal type size is mismatched");

struct xtree_leaf_node_header: qtree_node_header<unified_pointer<xtree_switch_node*> > { // for LGF
};
static_assert(sizeof(xtree_leaf_node_header) == 44, "internal type size is mismatched");

struct xtree_internal_node : xtree_node_header, qtree_internal_node<_xtree_node_ptr> {
};

struct xtree_switch_node : xtree_node_header, qtree_switch_node<_xtree_leaf_ptr> { // for LGF
};

struct xtree_leaf_node : xtree_leaf_node_header, qtree_leaf_node_metadata {
};

struct xtree_disk_node {
    union {
        xtree_node_header       prop;  // 44-byte
        xtree_leaf_node_header lprop;  // 44-byte
        xtree_internal_node    _intn;  // 44 + 40-byte
        xtree_switch_node      _swch;  // 44 + 8-byte // for LGF
        xtree_leaf_node        _leaf;  // 44 + 80-byte
        //xtree_leaf_list _leaf_list;
    }; // 124-byte

    xtree_disk_node() {
        static_assert(sizeof(xtree_leaf_node) == sizeof(xtree_disk_node));
        memset(&_leaf, 0, sizeof(xtree_leaf_node));
    }

    ASH_FORCEINLINE operator xtree_internal_node& () {
        return _intn;
    }

    ASH_FORCEINLINE operator xtree_internal_node const& () const {
        return _intn;
    }

    ASH_FORCEINLINE operator xtree_switch_node& () { // for LGF
        return _swch;
    }

    ASH_FORCEINLINE operator xtree_switch_node const& () const { // for LGF
        return _swch;
    }

    ASH_FORCEINLINE operator xtree_leaf_node& () {
        return _leaf;
    }

    ASH_FORCEINLINE operator xtree_leaf_node const& () const {
        return _leaf;
    }
};
#pragma pack(pop)

class disk_index_tree : ash::noncopyable {
public:
    disk_index_tree();

    disk_index_tree(
        gbid_t const& gbid,
        xtree_disk_node*  root_node,
        qtree_node_type   root_type,
        uint64_t      num_nodes,
        bool          enable_gc = true);

    disk_index_tree(
        gbid_t const& gbid,
        xtree_switch_node*  root_node,
        uint64_t      num_nodes,
        bool          enable_gc = true);

    disk_index_tree(
        gbid_t const& gbid,
        xtree_leaf_node* root_node,
        uint64_t     num_nodes,
        bool         enable_gc = true);

    void init(
        gbid_t const& gbid,
        xtree_disk_node* root_node,
        qtree_node_type   root_type,
        uint64_t      num_nodes,
        bool          enable_gc = true);

    disk_index_tree(disk_index_tree&& other) noexcept;
    disk_index_tree& operator=(disk_index_tree&& rhs) noexcept;
    ~disk_index_tree() noexcept;
    void post_order_dfs(std::function<void(xtree_disk_node*)> const& visit) const;

    qtree_node_type root_type() const {
        return _root_type;
    }

    xtree_disk_node* root_node() const {
        return _root_node;
    }

    uint64_t num_nodes() const {
        return _num_nodes;
    }

    gbid_t const& gbid() const {
        return _gbid;
    }

    bool is_null() const {
        return _root_node == nullptr;
    }

    static void print_node_info(xtree_internal_node* intn);
    static void print_node_info(xtree_leaf_node* leaf);
private:
    void _destroy_tree();

    gbid_t _gbid;
    xtree_disk_node* _root_node;
    uint64_t _num_nodes;
    bool _gc_enabled;
    qtree_node_type _root_type;
}; // class index_tree

#pragma pack(push, 8)
struct xtree_physical_pointer {
    gbid_t gbid;
    uint32_t num_nodes;
    uint64_t xtree_beg;
};
#pragma pack(pop)

} // namespace grid_format

} // namespace gstream
#endif // GSTREAM_GRID_FORMAT_DETAIL_DISK_INDEX_TREE_H
