#ifndef GSTREAM_FRAMEWORK3_DETAIL_LRU_CACHE_POLICY_H
#define GSTREAM_FRAMEWORK3_DETAIL_LRU_CACHE_POLICY_H
#include <ash/pooling_list.h>
#include <unordered_map>

namespace gstream {
namespace framework3 {
namespace detail {

template <typename Key>
class lru_cache_policy {
public:
    using key_type = Key;

    lru_cache_policy(size_t const pool_size = 256) :
        _node_pool(pool_size),
        _free_list(_node_pool),
        _lock_list(_node_pool) {

    }

    void add(key_type const& k) {
        assert(!test(k));
        _ptr_map.emplace(k, inverted_pointer{ _free_list.emplace_back(k).node(), false, 0 });
    }

    key_type evict() {
        assert(!_free_list.empty());
        auto node = _free_list.begin().node();
        key_type  key = node->value;
        assert(test(key));
        assert(!is_locked(key));
        _free_list.remove_node(node);
        _ptr_map.erase(key);
        return key;
    }

    unsigned lock(key_type const& k) {
        auto& inv_ptr = _ptr_map[k];
        if (is_locked(k)) {
            inv_ptr.ref_count += 1;
            return inv_ptr.ref_count;
        }
        _free_list.remove_node(inv_ptr.node);
        inv_ptr.is_locked = true;
        inv_ptr.ref_count = 1;
        inv_ptr.node = _lock_list.emplace_front(k).node();
        return 1;
    }

    unsigned unlock(key_type const& k) {
        assert(is_locked(k));
        auto& inv_ptr = _ptr_map[k];
        assert(inv_ptr.ref_count > 0);
        inv_ptr.ref_count -= 1;
        if (inv_ptr.ref_count == 0) {
            _lock_list.remove_node(inv_ptr.node);
            inv_ptr.is_locked = false;
            inv_ptr.node = _free_list.emplace_back(k).node();
        }
        return inv_ptr.ref_count;
    }

    void move_back(key_type const& k) { // for LGF
        auto& inv_ptr = _ptr_map[k];
        _free_list.remove_node(inv_ptr.node);
        inv_ptr.node = _free_list.emplace_back(k).node();
    }

    bool is_locked(key_type const& k) const {
        assert(test(k));
        return _ptr_map.at(k).is_locked;
    }

    bool test(key_type const& k) const {
        auto iter = _ptr_map.find(k);
        return iter != _ptr_map.cend();
    }

    size_t size() const {
        return _free_list.size();
    }

    void clear() {
        _free_list.clear();
        _lock_list.clear();
        _ptr_map.clear();
    }

    size_t num_locked() const {
        return _lock_list.size();
    }

    size_t num_unlocked() const {
        return _free_list.size();
    }

    size_t num_cached() const {
        return _ptr_map.size();
    }

private:
    using list_type = ash::pooling_list<key_type>;
    struct inverted_pointer {
        typename list_type::node_type* node;
        bool is_locked;
        unsigned ref_count;
    };
    typename list_type::pool_type _node_pool;
    list_type _free_list;
    list_type _lock_list;
    std::unordered_map<key_type, inverted_pointer> _ptr_map;
};

} // namespace detail
} // namespace framework3
} // namespace gstream

#endif // !GSTREAM_FRAMEWORK3_DETAIL_LRU_CACHE_POLICY_H
