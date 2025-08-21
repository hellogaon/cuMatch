#ifndef GSTREAM_GRID_FORMAT_DETAIL_INPUT_TREE_H
#define GSTREAM_GRID_FORMAT_DETAIL_INPUT_TREE_H
#include <gstream/grid_format/grid_format_defines.h>
#include <ash/detail/noncopyable.h>
#include <assert.h>
#include <string>
#include <functional>

namespace gstream {
namespace grid_format {
namespace detail {
namespace gridgen {

class input_tree: ash::noncopyable {
public:
    struct internal_node_t {
        //! [Mark: qtree-traversal-order]
        void* nw;
        void* sw;
        void* ne;
        void* se;
        qtree_node_type nw_type;
        qtree_node_type sw_type;
        qtree_node_type ne_type;
        qtree_node_type se_type;
        uint16_t level;

        void*& ptr(unsigned i, unsigned j) {
            //! [Mark: qtree-traversal-order]
            assert(i < 2 && "index overflow error.");
            assert(j < 2 && "index overflow error.");
            j <<= 1;
            i |= j;
            return reinterpret_cast<void**>(this)[i];
        }

        qtree_node_type& type(unsigned i, unsigned j) {
            //! [Mark: qtree-traversal-order]
            assert(i < 2 && "index overflow error.");
            assert(j < 2 && "index overflow error.");
            j <<= 1;
            i |= j;
            return (&nw_type)[i];
        }
    };

    struct leaf_node_t {
        std::string path; // disk
        e32_t* input_edges_ptr; // in-memory
        uint64_t block_size;
        bool is_temporal;
        bool is_big_endian;
        uint16_t level;
    };

    input_tree();
    input_tree(gbid_t const& gbid, void* root_node, qtree_node_type root_type, uint64_t num_nodes);
    input_tree(input_tree&& other) noexcept;
    input_tree& operator=(input_tree&& rhs) noexcept;
    ~input_tree() noexcept;
    void post_order_dfs(std::function<void(void*, qtree_node_type)> const& visit);

    qtree_node_type root_type() const {
        return _root_type;
    }

    void* root_node() const {
        return _root_node;
    }

    gbid_t const& gbid() const {
        return _gbid;
    }

    uint64_t num_nodes() const {
        return _num_nodes;
    }

    static void print_node_info(internal_node_t* intn);
    static void print_node_info(leaf_node_t* leaf);

private:
    void _destroy_tree();

    gbid_t _gbid;
    void* _root_node;
    qtree_node_type _root_type;
    uint64_t _num_nodes;
}; // class input_tree

} // namespace gridgen
} // namespace detail
} // namespace grid_format
} // namespace gstream
#endif // GSTREAM_GRID_FORMAT_DETAIL_INPUT_TREE_H
