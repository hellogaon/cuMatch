#ifndef CU_MATCH_CACHE_CONTROLLER_H
#define CU_MATCH_CACHE_CONTROLLER_H
#include <cu_match/subgraph_match_defines.h>
#include <gstream/framework3/detail/lru_cache_policy.h>
#include <gstream/grid_format/grid_format_defines.h>
#include <ash/utility/dbg_log.h>

namespace cu_match {

template <typename KeyTy, typename ValueTy>
class lru_cache_controller {
public:
    using key_type = KeyTy;
    using value_type = ValueTy;
    static_assert(std::is_integral_v<key_type>, "Key type must be integral type");

    struct entity_type {
        value_type value;
        bool valid : 1;
        bool _reserved : 7;
    };

    struct evict_result {
        bool success;
        key_type key;
        value_type value;
    };

    struct config_t {
        uint64_t num_entities;
    };

    lru_cache_controller() {
        memset(&_cfg, 0, sizeof(config_t));
        _container = nullptr;
        _lru = nullptr;
    }

    ~lru_cache_controller() noexcept {
        close();
    }

    [[nodiscard]] bool init(config_t const& cfg) noexcept {
        if (cfg.num_entities == 0)
            return false;

        _cfg = cfg;
        entity_type* container = static_cast<entity_type*>(calloc(_cfg.num_entities, sizeof(entity_type)));
        cache_policy* lru = nullptr;
        if (container == nullptr)
            goto lb_error;
        try {
            lru = new cache_policy(std::max(_cfg.num_entities / 16, (uint64_t)128));
        }
        ASH_CLAUSE_CATCH_STDEXCEPT_WITH(goto lb_error);
        
        _container = container;
        _lru = lru;
        return true;

    lb_error:
        free(container);
        delete(lru);
        return false;
    }

    void close() noexcept {
        delete _lru; _lru = nullptr;
        free(_container); _container = nullptr;
    }

    [[nodiscard]] bool is_initialized() const noexcept {
        return _container != nullptr;
    }

    [[nodiscard]] value_type const* get(key_type const& key) const noexcept {
        assert(is_initialized());
        assert(key < _cfg.num_entities);
        if (ASH_LIKELY(_container[key].valid))
            return &_container[key].value;
        return nullptr;
    }

    [[nodiscard]] value_type* get(key_type const& key) noexcept {
        _lru->move_back(key);
        return const_cast<value_type*>(std::as_const(*this).get(key));
    }

    [[nodiscard]] value_type const* operator[](key_type const& key) const noexcept {
        if (ASH_LIKELY(_container[key].valid))
            return &_container[key].value;
        return nullptr;
    }

    [[nodiscard]] value_type* operator[](key_type const& key) noexcept {
        return const_cast<value_type*>(std::as_const(*this).operator[](key));
    }

    value_type* add(key_type const& key, value_type const& val) {
        assert(is_initialized());
        assert(key < _cfg.num_entities);
        entity_type& entity = _container[key];
        assert(!entity.valid);
        entity.value = val;
        entity.valid = true;
        _lru->add(key);
        return &entity.value;
    }

    [[nodiscard]] evict_result evict() {
        evict_result r;
        if (num_unlocked() == 0) {
            r.success = false;
            return r;
        }
        r.success = true;
        key_type const key = _lru->evict();
        r.key = key;
        r.value = std::move(_container[key].value);
        _container[key].valid = false;
        return r;
    }

    [[nodiscard]] bool test(key_type const& key) const noexcept {
        assert(is_initialized());
        assert(key < _cfg.num_entities);
        return _container[key].valid;
    }

    unsigned lock(key_type const& key) {
        assert(test(key));
        return _lru->lock(key); // returns current reference count
    }

    unsigned unlock(key_type const& key) {
        assert(test(key));
        return _lru->unlock(key); // returns current reference count
    }

    uint64_t num_locked() const noexcept {
        return _lru->num_locked();
    }

    uint64_t num_unlocked() const noexcept {
        return _lru->num_unlocked();
    }

    uint64_t num_cached() const noexcept {
        return _lru->num_cached();
    }

    config_t const& config() const noexcept {
        return _cfg;
    }

private:
    using cache_policy = gstream::framework3::detail::lru_cache_policy<key_type>;

    config_t _cfg;
    entity_type* _container;
    cache_policy* _lru;
};

struct shard_descriptor {
    void* addr;
    uint64_t size;
};

class shard_cachectl: public lru_cache_controller<gstream::grid_format::shard_uid, shard_descriptor> {
};

} // namespace cu_match

#endif // CU_MATCH_CACHE_CONTROLLER_H
