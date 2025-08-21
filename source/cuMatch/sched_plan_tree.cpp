#include <cu_match/sched_plan_tree.h>
#include <stack>

namespace cu_match {

sched_plan_tree::sched_plan_tree() {}

sched_plan_tree::~sched_plan_tree() {
    _cleanup();
}

void sched_plan_tree::init(config_t const& cfg) {
    _cfg = cfg;
    _root_node = _cfg.root_node;
    _iterate_init();
}

void sched_plan_tree::pre_order_dfs(std::function<void(sched_plan_node*)> const& visit) {
    std::vector<sched_plan_node*> visit_vec;
    visit_vec.push_back(_root_node);

    _backtracking(_root_node, visit_vec);

    for(sched_plan_node* p: visit_vec)
        visit(p);
}

void sched_plan_tree::post_order_dfs(std::function<void(sched_plan_node*)> const& visit) {
    std::stack<sched_plan_node*> visit_stk;
    std::stack<sched_plan_node*> traversal_stk;
    traversal_stk.push(_root_node);

    while (!traversal_stk.empty()) {
        sched_plan_node* node = traversal_stk.top();
        traversal_stk.pop();
        visit_stk.push(node);
        for (sched_plan_node* p: node->child) {
            traversal_stk.push(p);
        }
    }

    while (!visit_stk.empty()) {
        sched_plan_node* p = visit_stk.top();
        visit_stk.pop();
        visit(p);
    }
}

bool sched_plan_tree::pop_grid_group() {
    if (_this_node() == nullptr) {
        _grid_group_ptr = nullptr;
        return false;
    }
    while (true) {
        sched_plan_node* node = _next_node();
        
        if (node == nullptr) {
            grid_group_info* g_info = new grid_group_info();
            *g_info = _tmp_group_info;
            _grid_group_ptr = g_info;
            return true;
        }

        if (node->level == 1 || _tmp_group_info.v_id != node->v_id || _tmp_group_info.max_level >= node->level || _cfg.is_table_join) {
            if (_tmp_group_info.v_id) { // not 0, 0 means root node
                // return grid group
                grid_group_info* g_info = new grid_group_info();
                *g_info = _tmp_group_info;
                _grid_group_ptr = g_info;
                
                // make new grid group
                _tmp_group_info.v_id = node->v_id;
                _tmp_group_info.min_level = node->level;
                _tmp_group_info.max_level = node->level;
                _tmp_group_info.grid_group.clear();
                _tmp_group_info.grid_group.push_back(node->leaf_ptr);

                _tmp_group_info.st_info = nullptr;
                if (node->level == _cfg.max_level) {
                    assert(_tracking_stk.size() == _cfg.max_level + 1);
                    _tmp_group_info.st_info = make_subtask_info();
                }
                return true;
            }
            _tmp_group_info.v_id = node->v_id;
            _tmp_group_info.min_level = node->level;
            _tmp_group_info.max_level = node->level;
            _tmp_group_info.grid_group.clear();
            _tmp_group_info.st_info = nullptr;
        }
        _tmp_group_info.max_level = node->level;
        // update grid group
        _tmp_group_info.grid_group.push_back(node->leaf_ptr);
        if (node->level == _cfg.max_level) {
            assert(_tracking_stk.size() == _cfg.max_level + 1);
            _tmp_group_info.st_info = make_subtask_info();
        }
    };
    return true;
}

uint64_t sched_plan_tree::pop_single_kernel_task(single_kernel_task_info* k_info, uint64_t capacity, bool initial) {
    uint64_t used = 0;
    _cache.clear();
    _tree_info_vec.clear();

    if (!initial) {
        if (_grid_group_ptr == nullptr) {
            return 0;
        }

        // add first grid group (default)
        for (uint32_t i = 1; i <= _convert_elevel_to_vlevel(_grid_group_ptr->max_level); i++) {
            _tree_info_vec.push_back({ i, 0 });
        }

        if (_grid_group_ptr->st_info != nullptr) {
            for (uint32_t i = 0; i < _cfg.max_level; i++) {
                sleaf_ptr const sleaf = _grid_group_ptr->st_info->sleaf_ptrs[i];
                uint64_t u_id = sleaf->unique_id;
                if (_cache.find(u_id) == _cache.end()) {
                    _cache.insert(u_id);
                    used += sleaf->size_info.in_memory_size;
                    k_info->total_sleaf_ptrs.push_back(sleaf);
                }
            }
            k_info->subtasks.push_back(_grid_group_ptr->st_info);
        }
        assert(used < capacity); // have to increase GPU memory
    }

    while (true) {
        bool rslt = pop_grid_group();

        if (rslt == false)
            break;

        uint32_t required_size = 0;
        if (_grid_group_ptr->st_info != nullptr) {
            for (uint32_t i = 0; i < _cfg.max_level; i++) {
                sleaf_ptr const sleaf = _grid_group_ptr->st_info->sleaf_ptrs[i];
                uint64_t u_id = sleaf->unique_id;
                if (_cache.find(u_id) == _cache.end()) {
                    required_size += sleaf->size_info.in_memory_size;
                }
            }
        }

        // heuristic algorithm
        if ((required_size + used >= capacity) 
                || (_convert_elevel_to_vlevel(_grid_group_ptr->min_level) == 1 && ((used > capacity / 10 * 9) || (_cache.size() > MaxNumShardPerIter / 10 * 9) || (_cfg.num_query_vertices * k_info->subtasks.size() + 1 > MaxNumTreeNodes / 10 * 9)))) {
            if (_cfg.num_optional_vertices > 0)
                assert(_convert_elevel_to_vlevel(_grid_group_ptr->min_level) == 1);
            assert(used != 0); // have to increase GPU memory
            break;
        }
        
        // add grid group
        _tree_info_vec.push_back({ _convert_elevel_to_vlevel(_grid_group_ptr->max_level), static_cast<uint32_t>(k_info->subtasks.size()) });
        used += required_size;

        if (_grid_group_ptr->st_info != nullptr) {
            for (uint32_t i = 0; i < _cfg.max_level; i++) {
                sleaf_ptr const sleaf = _grid_group_ptr->st_info->sleaf_ptrs[i];
                uint64_t u_id = sleaf->unique_id;
                if (_cache.find(u_id) == _cache.end()) {
                    _cache.insert(u_id);
                    k_info->total_sleaf_ptrs.push_back(sleaf);
                }
            }
            k_info->subtasks.push_back(_grid_group_ptr->st_info);
        }

        delete _grid_group_ptr;
        _grid_group_ptr = nullptr;
    }

    if (!_cfg.is_table_join) _regenerate_LFTJ_grid_group_tree(k_info);
    else _regenerate_MJ_grid_group_tree(k_info);
    return used;
}

void sched_plan_tree::print_sched_plan() {
    this->pre_order_dfs([&](sched_plan_node* node) {
        if (node == _root_node)
            return;
        for (uint16_t i = 0; i < node->level; i++)
            printf("\t");

        sleaf_ptr& sleaf = node->leaf_ptr;
        gbid_t gbid = sleaf->gbid;
        printf("%lu: (%u, %u)\n", sleaf->unique_id, gbid.row, gbid.col);
    });
}

void sched_plan_tree::_cleanup() {
    this->post_order_dfs([&](sched_plan_node* node) {
        delete node;
    });
    _root_node = nullptr;
}

void sched_plan_tree::_iterate_init() {
    _tracking_stk.reserve(_cfg.max_level);
    _tracking_stk.clear();
    _tracking_stk.push_back(tracking_info{ _root_node, 0 });
    _grid_group_ptr = nullptr;
    _tmp_group_info = grid_group_info{0};
    _cache.clear();
}

void sched_plan_tree::_backtracking(sched_plan_node* node, std::vector<sched_plan_node*>& rslt) {
    if (node->child.size() == 0) return;
    for (uint64_t i = 0; i < node->child.size(); i++) {
        sched_plan_node* p = (node->child)[i];
        rslt.push_back(p);
        _backtracking(p, rslt);
    }
}

sched_plan_node* sched_plan_tree::_this_node() {
    if (_tracking_stk.empty())
        return nullptr;
    tracking_info t_info_top = _tracking_stk.back();
    return t_info_top.node;
}

sched_plan_node* sched_plan_tree::_next_node() {
    if (_tracking_stk.empty())
        return nullptr;
    tracking_info t_info_top = _tracking_stk.back();
    while (t_info_top.node->child.size() <= t_info_top.child_idx) {
        _tracking_stk.pop_back();
        if (_tracking_stk.empty())
            break;
        t_info_top = _tracking_stk.back();
    }
    if (_tracking_stk.empty())
        return nullptr;
    _tracking_stk.back().child_idx++;
    _tracking_stk.push_back(tracking_info{ t_info_top.node->child[t_info_top.child_idx], 0 });
    return _this_node();
}

subtask_info* sched_plan_tree::make_subtask_info() {
    subtask_info* st_info = new subtask_info;
    st_info->sleaf_ptrs = static_cast<sleaf_ptr*>(calloc(sizeof(sleaf_ptr), _cfg.max_level));

    for (uint32_t i = 1; i < _tracking_stk.size(); i++) {
        st_info->sleaf_ptrs[i - 1] = _tracking_stk[i].node->leaf_ptr;
    }
    return st_info;
}

void sched_plan_tree::_regenerate_LFTJ_grid_group_tree(single_kernel_task_info* k_info) {
    struct stack_element {
        uint32_t v_id;
        uint32_t dep;
    };
    struct tree_element {
        uint32_t node_id;
        uint32_t st_id;
    };
    std::vector<std::vector<tree_element> > tree;
    
    uint32_t const num_regular_vertices = _cfg.num_query_vertices - _cfg.num_optional_vertices;
    uint32_t const max_regular_vid = num_regular_vertices - _cfg.postfix_order->size();
    uint32_t const max_optional_vid = _cfg.num_query_vertices - _cfg.optional_postfix_order->size();
    uint32_t const max_num_tree_nodes = _cfg.num_query_vertices * k_info->subtasks.size() + 1;
    tree.resize(max_num_tree_nodes);
    std::stack<stack_element> stk;

    uint32_t tree_node_id = 1, child_num = 0;
    stk.push({0, 0});
    for (uint32_t i = 0; i < _tree_info_vec.size(); i++) {
        uint32_t const dep = _tree_info_vec[i].v_level, st_id = _tree_info_vec[i].subtask_id;
        if (st_id >= k_info->subtasks.size()) break; 
        while (stk.top().dep >= dep) stk.pop();
        tree[stk.top().v_id].push_back({tree_node_id, st_id});
        if (dep == 1) k_info->tree_root_id.push_back(tree_node_id);
        stk.push({tree_node_id++, dep});
        child_num++;
        if (stk.top().dep == max_regular_vid) {
            for (uint32_t j = 0; j < _cfg.postfix_order->size(); j++) {
                tree[stk.top().v_id].push_back({tree_node_id, st_id});
                stk.push({tree_node_id++, dep});
                child_num++;
            }
        }
        if (stk.top().dep == max_optional_vid) {
            for (uint32_t j = 0; j < _cfg.optional_postfix_order->size(); j++) {
                tree[stk.top().v_id].push_back({tree_node_id, st_id});
                stk.push({tree_node_id++, dep});
                child_num++;
            }
        }
    }

    assert(child_num == tree_node_id - 1); // because of tree
    
    // for (uint32_t i = 0; i < tree_node_id; i++) {
    //     printf("%u : [ ", i);
    //     for(uint32_t j = 0; j < tree[i].size(); j++)
    //         printf("%u ", tree[i][j]);
    //     printf("]\n");
    // }

    k_info->num_tree_nodes = tree_node_id;
    k_info->sched_tree_ptr = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes + 1));
    k_info->sched_tree_nxt_id = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes - 1));
    k_info->sched_tree_prv_id = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes - 1));
    k_info->sched_tree_st_id = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes - 1));

    uint32_t sum = 0;
    uint32_t dst_idx = 0;
    for (uint32_t i = 0; i < tree_node_id; i++) {
        k_info->sched_tree_ptr[i] = sum;
        sum += tree[i].size();
        for (uint32_t j = 0; j < tree[i].size(); j++) {
            k_info->sched_tree_nxt_id[dst_idx] = tree[i][j].node_id;
            k_info->sched_tree_prv_id[tree[i][j].node_id - 1] = i;
            k_info->sched_tree_st_id[dst_idx++] = tree[i][j].st_id;
        }
    }
    k_info->sched_tree_ptr[k_info->num_tree_nodes] = sum;

    if (k_info->num_tree_nodes >= MaxNumTreeNodes) {
        printf("Constant MaxNumTreeNodes must be greater than %u\n", k_info->num_tree_nodes);
        exit(1);
    }
}

void sched_plan_tree::_regenerate_MJ_grid_group_tree(single_kernel_task_info* k_info) {
    struct stack_element {
        uint32_t e_id;
        uint32_t dep;
    };
    struct tree_element {
        uint32_t node_id;
        uint32_t st_id;
    };
    std::vector<std::vector<tree_element> > tree;
    
    uint32_t const max_num_tree_nodes = _cfg.num_query_edges * k_info->subtasks.size() + 1;
    tree.resize(max_num_tree_nodes);
    std::stack<stack_element> stk;

    uint32_t tree_node_id = 1, child_num = 0;
    stk.push({0, 0});
    for (uint32_t i = 0; i < _tree_info_vec.size(); i++) {
        uint32_t const dep = _tree_info_vec[i].v_level, st_id = _tree_info_vec[i].subtask_id;
        if (st_id >= k_info->subtasks.size()) break; 
        while (stk.top().dep >= dep) stk.pop();
        tree[stk.top().e_id].push_back({tree_node_id, st_id});
        if (dep == 1) k_info->tree_root_id.push_back(tree_node_id);
        stk.push({tree_node_id++, dep});
        child_num++;
    }

    assert(child_num == tree_node_id - 1); // because of tree

    // for (uint32_t i = 0; i < tree_node_id; i++) {
    //     printf("%u : [ ", i);
    //     for(uint32_t j = 0; j < tree[i].size(); j++)
    //         printf("%u ", tree[i][j]);
    //     printf("]\n");
    // }

    k_info->num_tree_nodes = tree_node_id;
    k_info->sched_tree_ptr = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes + 1));
    k_info->sched_tree_nxt_id = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes - 1));
    k_info->sched_tree_prv_id = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes - 1));
    k_info->sched_tree_st_id = static_cast<uint32_t*>(calloc(sizeof(uint32_t), k_info->num_tree_nodes - 1));

    uint32_t sum = 0;
    uint32_t dst_idx = 0;
    for (uint32_t i = 0; i < tree_node_id; i++) {
        k_info->sched_tree_ptr[i] = sum;
        sum += tree[i].size();
        for (uint32_t j = 0; j < tree[i].size(); j++) {
            k_info->sched_tree_nxt_id[dst_idx] = tree[i][j].node_id;
            k_info->sched_tree_prv_id[tree[i][j].node_id - 1] = i;
            k_info->sched_tree_st_id[dst_idx++] = tree[i][j].st_id;
        }
    }
    k_info->sched_tree_ptr[k_info->num_tree_nodes] = sum;

    if (k_info->num_tree_nodes >= MaxNumTreeNodes) {
        printf("Constant MaxNumTreeNodes must be greater than %u\n", k_info->num_tree_nodes);
        exit(1);
    }
}

uint32_t sched_plan_tree::_convert_elevel_to_vlevel(uint32_t level) { 
    return (*_cfg.elevel_to_vlevel)[level - 1] + 1;
}

} // namespace cu_match
