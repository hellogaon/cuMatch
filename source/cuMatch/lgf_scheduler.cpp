#include <cu_match/lgf_scheduler.h>
#include <gstream/grid_format/detail/gridgen_utility.h>
#include <stack>

namespace cu_match {

lgf_scheduler::lgf_scheduler() {}

lgf_scheduler::~lgf_scheduler() {
    _cleanup();
}

void lgf_scheduler::init(config_t const& cfg) {
    _cfg = cfg;
    _num_vertices = _cfg.qgraph->num_query_vertices();
    _num_edges = _cfg.qgraph->num_query_edges();
    _num_constraints = _cfg.qgraph->num_constraints();
    _num_optional_vertices = _cfg.qgraph->num_optional_vertices();
    _vertices_info = _cfg.qgraph->qgraph_vertices_info();
    _edges_info = _cfg.qgraph->qgraph_edges_info();
    _constraint_info = _cfg.qgraph->qgraph_constraints_info();
    _order = static_cast<order_element*>(calloc(_num_vertices, sizeof(order_element)));
    _order_rank.resize(_num_vertices);
    _adj_list.resize(_num_vertices);
    _edge_order.resize(_num_edges);
    _sleaf_rslt.resize(_num_edges);
    _range_rslt.resize(_num_vertices);
    _elevel_to_vlevel.resize(_num_edges);
    for (uint32_t i = 0; i < _num_vertices; i++) {
        _range_rslt[i].local_range.push(gstream::grid_format::laddr_interval{ 0, gstream::grid_format::GridBlockWidth - 1 });
    }
}

bool lgf_scheduler::make_LGF_schedule(order_element* order, order_element* edge_order) {
    if (!_cfg.is_table_join) {
        return _make_LFTJ_schedule(order, edge_order);
    }
    else {
        return _make_MJ_schedule(order, edge_order);
    }
}

// order by (regular intersection vertices) + (regular postfix_order_vertices) + (post intersection vertices) + (post postfix_order_vertices)
bool lgf_scheduler::_make_LFTJ_schedule(order_element* order, order_element* /*edge_order*/) {
    // reordering for LFTJ
    for (uint32_t i = 0; i < _num_vertices; i++) {
        _order_rank[order[i]] = i;
    }
    for (uint32_t i = 0 ; i < _num_edges; i++) {
        uint32_t const v1_id = _edges_info[i].v1_id;
        uint32_t const v2_id = _edges_info[i].v2_id;
        uint32_t v1_rank = _order_rank[v1_id];
        uint32_t v2_rank = _order_rank[v2_id];
        assert(v1_rank != v2_rank);
        if (_edges_info[i].optional_edge_with_regular_vertex) {
            if (!_vertices_info[v1_id].is_optional)
                _adj_list[v2_id].push_back( {i, order[v1_id]} );
            else if (!_vertices_info[v2_id].is_optional)
                _adj_list[v1_id].push_back( {i, order[v2_id]} );
            else {
                if (v1_rank < v2_rank)
                    _adj_list[v1_id].push_back( {i, order[v2_id]} );
                else 
                    _adj_list[v2_id].push_back( {i, order[v1_id]} );
            }
        }
        else {
            if (v1_rank < v2_rank)
                _adj_list[v1_id].push_back( {i, order[v2_id]} );
            else 
                _adj_list[v2_id].push_back( {i, order[v1_id]} );
        }
    }
    for (uint32_t i = 0; i < _num_vertices; i++)
        std::sort(_adj_list[i].begin(), _adj_list[i].end(), [](const _adj_list_info& a, const _adj_list_info& b) {
            return a.opposite_order_id < b.opposite_order_id;
        });
    
    // preprocessing (make new order: split postfix order)
    uint32_t e_id = 0, order_idx = 0;
    std::vector<uint32_t> optional_vertices;
    for (uint32_t i = 0; i < _num_vertices; i++) {
        gstream::grid_off qv_id = order[i];
        if (_adj_list[qv_id].empty()) {
            if (!_vertices_info[qv_id].is_optional)
                _postfix_order.push_back(qv_id);                
            else
                _optional_postfix_order.push_back(qv_id);
        }
        else {
            if (!_vertices_info[qv_id].is_optional) {
                _order[order_idx] = qv_id;
                for (uint32_t j = 0; j < _adj_list[qv_id].size(); j++) {
                    _edge_order[e_id] = { _adj_list[qv_id][j].edge_id, order_idx };
                    _elevel_to_vlevel[e_id] = order_idx;
                    e_id++;
                }
                order_idx++;
            }
            else {
                optional_vertices.push_back(qv_id);
            }
        }
    }
    for (uint32_t i = 0; i < _postfix_order.size(); i++)
        _order[order_idx++] = _postfix_order[i];
    for (uint32_t i = 0; i < optional_vertices.size(); i++) {
        gstream::grid_off qv_id = optional_vertices[i];
        _order[order_idx] = qv_id;
        for (uint32_t j = 0; j < _adj_list[qv_id].size(); j++) {
            _edge_order[e_id] = { _adj_list[qv_id][j].edge_id, order_idx };
            _elevel_to_vlevel[e_id] = order_idx;
            e_id++;
        }
        order_idx++;
    }
    for (uint32_t i = 0; i < _optional_postfix_order.size(); i++)
        _order[order_idx++] = _optional_postfix_order[i];
    
    assert(order_idx == _num_vertices);
    
    for (uint32_t i = 0; i < _num_vertices; i++) 
        _order_rank[_order[i]] = i;


    sched_plan_node* root = _make_sched_plan();
    
    // post-processing
    sched_plan_tree::config_t spt_cfg;
    spt_cfg.root_node = root;
    spt_cfg.max_level = _num_edges;
    spt_cfg.num_query_vertices = _num_vertices;
    spt_cfg.num_optional_vertices = _num_optional_vertices;
    spt_cfg.postfix_order = &_postfix_order;
    spt_cfg.optional_postfix_order = &_optional_postfix_order;
    spt_cfg.elevel_to_vlevel = &_elevel_to_vlevel;
    spt_cfg.is_table_join = _cfg.is_table_join;
    _sched_plan_tree.init(spt_cfg);

#ifdef DEBUG
    _print_scheduler();
#endif
    return true;
}

bool lgf_scheduler::_make_MJ_schedule(order_element* /*order*/, order_element* edge_order) {
    assert(_num_edges > 1);
    bool visited[MaxNumQueryVertices] = {};
    uint32_t order_idx = 0;
    for (uint32_t i = 0 ; i < _num_edges; i++) {
        uint32_t const e_id = edge_order[i];
        uint32_t const v1_id = _edges_info[e_id].v1_id;
        uint32_t const v2_id = _edges_info[e_id].v2_id;
        assert(v1_id != v2_id);
        if (!visited[v1_id]) {
            _order[order_idx++] = v1_id;
        }
        if (!visited[v2_id]) {
            _order[order_idx++] = v2_id;
        }
        visited[v1_id] = true;
        visited[v2_id] = true;
    }

    for (uint32_t i = 0; i < _num_vertices; i++) {
        _order_rank[_order[i]] = i;
    }

    for (uint32_t i = 0 ; i < _num_edges; i++) {
        uint32_t const e_id = edge_order[i];
        uint32_t const v1_id = _edges_info[e_id].v1_id;
        uint32_t const v2_id = _edges_info[e_id].v2_id;
        uint32_t v1_rank = _order_rank[v1_id];
        uint32_t v2_rank = _order_rank[v2_id];
        assert(v1_rank != v2_rank);
        if (v1_rank < v2_rank) {
            _adj_list[v1_id].push_back( {i, _order[v2_id]} );
            _edge_order[i] = { e_id, _order_rank[v1_id] };
        }
        else {
            _adj_list[v2_id].push_back( {i, _order[v1_id]} );
            _edge_order[i] = { e_id, _order_rank[v2_id] };
        }
        _elevel_to_vlevel[i] = i;
    }

    sched_plan_node* root = _make_sched_plan();
    
    // post-processing
    sched_plan_tree::config_t spt_cfg;
    spt_cfg.root_node = root;
    spt_cfg.max_level = _num_edges;
    spt_cfg.num_query_vertices = _num_vertices;
    spt_cfg.num_query_edges = _num_edges;
    spt_cfg.num_optional_vertices = _num_optional_vertices;
    spt_cfg.postfix_order = &_postfix_order;
    spt_cfg.optional_postfix_order = &_optional_postfix_order;
    spt_cfg.elevel_to_vlevel = &_elevel_to_vlevel;
    spt_cfg.is_table_join = _cfg.is_table_join;
    _sched_plan_tree.init(spt_cfg);

#ifdef DEBUG
    _print_scheduler();
#endif
    return true;
}

void lgf_scheduler::init_global_kernel_info(global_kernel_task_info* gk_info) {
    gk_info->global_order = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_vertices));
    gk_info->edge_type_list = static_cast<kernel_edge_type*>(calloc(sizeof(kernel_edge_type), _num_edges));
    gk_info->prefix_sum_degree = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_vertices + 1));
    gk_info->eid_list = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_edges * 2));
    gk_info->lv0_dep = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_edges));
    gk_info->v1_list_idxs = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_edges));
    gk_info->v2_list_idxs = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_edges));
    gk_info->constraint_ptr = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_vertices + 1));
    gk_info->constraints = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_constraints));
    gk_info->num_optional_per_vertex = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_vertices));
    gk_info->num_negative_per_vertex = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_vertices));
    gk_info->is_optional_vertex = static_cast<bool*>(calloc(sizeof(bool), _num_vertices));
    gk_info->is_negative_vertex_with_degree = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_vertices));
    // for table join
    gk_info->edge_order_with_vid = static_cast<uint32_t*>(calloc(sizeof(uint32_t), _num_edges * 2));
    gk_info->is_join_vertex = static_cast<bool*>(calloc(sizeof(bool), _num_vertices));

    {
        for (uint32_t i = 0; i < _num_vertices; i++) {
            gk_info->global_order[i] = _order[i];
        }
    }

    {
        uint32_t degree_sum = 0;
        for (uint32_t i = 0; i < _num_vertices; i++) {
            gk_info->prefix_sum_degree[i] = degree_sum;
            degree_sum += _vertices_info[i].degree;
        }
        gk_info->prefix_sum_degree[_num_vertices] = degree_sum;
    }
    
    {
        uint32_t degree_counter[MaxNumQueryVertices] = {};
        uint32_t prefix_sum_complex_edge[MaxNumQueryVertices] = {};
        uint32_t complex_edge_degree_counter[MaxNumQueryVertices] = {};

        for (uint32_t i = 0; i < _num_vertices; i++) {
            prefix_sum_complex_edge[i] = gk_info->prefix_sum_degree[i];
        }
        for (uint32_t i = 0; i < _num_edges; i++) {
            if (_edges_info[i].e_type == qgraph_edge_type::REGULAR) {
                uint32_t v1_id = _edges_info[i].v1_id;
                uint32_t v2_id = _edges_info[i].v2_id;
                prefix_sum_complex_edge[v1_id]++;
                prefix_sum_complex_edge[v2_id]++;
            }
        }

        for (uint32_t i = 0; i < _num_edges; i++) {
            uint32_t eid = _edge_order[i].edge_id;
            qgraph_edge_type e_ty = _edges_info[eid].e_type;
            uint32_t v1_id = _edges_info[eid].v1_id;
            uint32_t v2_id = _edges_info[eid].v2_id;
            uint32_t v1_list_idx, v2_list_idx;

            if (e_ty == qgraph_edge_type::REGULAR) {
                v1_list_idx = gk_info->prefix_sum_degree[v1_id] + degree_counter[v1_id];
                v2_list_idx = gk_info->prefix_sum_degree[v2_id] + degree_counter[v2_id];
                degree_counter[v1_id]++;
                degree_counter[v2_id]++;
            }
            else {
                v1_list_idx = prefix_sum_complex_edge[v1_id] + complex_edge_degree_counter[v1_id];
                v2_list_idx = prefix_sum_complex_edge[v2_id] + complex_edge_degree_counter[v2_id];
                complex_edge_degree_counter[v1_id]++;
                complex_edge_degree_counter[v2_id]++;
            }

            gk_info->eid_list[v1_list_idx] = (_order_rank[v1_id] < _order_rank[v2_id] ? eid : eid + _num_edges);
            gk_info->eid_list[v2_list_idx] = (_order_rank[v2_id] < _order_rank[v1_id] ? eid : eid + _num_edges);

            if (e_ty == qgraph_edge_type::REGULAR) {
                gk_info->edge_type_list[v1_list_idx] = kernel_edge_type::REGULAR;
                gk_info->edge_type_list[v2_list_idx] = kernel_edge_type::REGULAR;
            }
            else if (e_ty == qgraph_edge_type::OPTIONAL) {
                gk_info->edge_type_list[v1_list_idx] = kernel_edge_type::OPTIONAL;
                gk_info->edge_type_list[v2_list_idx] = kernel_edge_type::OPTIONAL;
                if (_order_rank[v1_id] < _order_rank[v2_id]) {
                    gk_info->num_optional_per_vertex[v1_id]++;
                }
                else {
                    gk_info->num_optional_per_vertex[v2_id]++;
                }
            }
            else if (e_ty == qgraph_edge_type::NEGATIVE) {
                gk_info->edge_type_list[v1_list_idx] = kernel_edge_type::NEGATIVE;
                gk_info->edge_type_list[v2_list_idx] = kernel_edge_type::NEGATIVE;
                if (_order_rank[v1_id] < _order_rank[v2_id]) {
                    gk_info->num_negative_per_vertex[v1_id]++;
                    gk_info->is_negative_vertex_with_degree[_order_rank[v2_id]]++;
                }
                else {
                    gk_info->num_negative_per_vertex[v2_id]++;
                    gk_info->is_negative_vertex_with_degree[_order_rank[v1_id]]++;
                }
            }
            
            gk_info->lv0_dep[eid] = std::min(_order_rank[v1_id], _order_rank[v2_id]);

            gk_info->v1_list_idxs[i] = v1_list_idx;
            gk_info->v2_list_idxs[i] = v2_list_idx;
        }
    }

    {   
        std::vector<uint32_t> const_adj[MaxNumQueryVertices];
        for (uint32_t i = 0; i < _num_constraints; i++) {
            uint32_t v1_id = _constraint_info[i].v1_id;
            uint32_t v2_id = _constraint_info[i].v2_id;
            qgraph_constraint_type c_type = _constraint_info[i].c_type;

            if (c_type == qgraph_constraint_type::NOT_EQUAL) {
                if (_order_rank[v1_id] < _order_rank[v2_id])
                    const_adj[v2_id].push_back(_order_rank[v1_id]);
                else 
                    const_adj[v1_id].push_back(_order_rank[v2_id]);
            }
        }
        
        uint32_t prefix_sum = 0, const_idx = 0;
        for (uint32_t i = 0; i < _num_vertices; i++) {
            gk_info->constraint_ptr[i] = prefix_sum;
            prefix_sum += const_adj[i].size();
            for (uint32_t j = 0; j < const_adj[i].size(); j++)
                gk_info->constraints[const_idx++] = const_adj[i][j];
        }
        gk_info->constraint_ptr[_num_vertices] = prefix_sum;
    }

    {
        for (uint32_t i = 0; i < _num_vertices; i++) {
            gk_info->is_optional_vertex[i] = _vertices_info[_order[i]].is_optional;
        }
    }

    // for table join
    {
        if (_cfg.is_table_join) {
            bool visited[MaxNumQueryVertices] = {};
            for (uint32_t i = 0; i < _num_edges; i++) {
                uint32_t eid = _edge_order[i].edge_id;
                uint32_t v1_id = _edges_info[eid].v1_id;
                uint32_t v2_id = _edges_info[eid].v2_id;
                if (_order_rank[v1_id] < _order_rank[v2_id]) {
                    gk_info->edge_order_with_vid[i * 2] = v1_id;
                    gk_info->edge_order_with_vid[i * 2 + 1] = v2_id;
                    if (visited[v1_id]) gk_info->is_join_vertex[i * 2] = true;
                    if (visited[v2_id]) gk_info->is_join_vertex[i * 2 + 1] = true;
                }
                else {
                    gk_info->edge_order_with_vid[i * 2] = v2_id;
                    gk_info->edge_order_with_vid[i * 2 + 1] = v1_id;
                    if (visited[v2_id]) gk_info->is_join_vertex[i * 2] = true;
                    if (visited[v1_id]) gk_info->is_join_vertex[i * 2 + 1] = true;
                }
                visited[v1_id] = visited[v2_id] = true;
            }
        }
    }
}

void lgf_scheduler::_cleanup() {
    free(_order); _order = nullptr;
}

// DFS traverse for find all sub-task
sched_plan_node* lgf_scheduler::_make_sched_plan() {
    sched_plan_node* root = new sched_plan_node();
    root->parent = nullptr;
    root->level = 0;
    root->v_id = 0;
    root->e_id = 0;
    root->leaf_ptr = nullptr;
    root->valid = false;

    _iterate_sub_task(0, root);
    
    assert(root->valid && "There are no matching grid combinations");

    return root;
}

void lgf_scheduler::_iterate_sub_task(uint32_t edge_idx, sched_plan_node* parent) {
    // found sub-task
    if (edge_idx >= _num_edges) {
        parent->valid = true;
        return;
    }

    using namespace gstream;
    using namespace grid_format;
    
    uint32_t e_id = _edge_order[edge_idx].edge_id;
    uint32_t odr = _edge_order[edge_idx].order;
    qgraph_edge_type e_ty = _edges_info[e_id].e_type;
    uint32_t qv_id = _order[odr];
    bool is_out_edges_shard = true; // if true, we have to load in-edges shard

    qgraph_edge_info& e_info = _edges_info[e_id];
    uint32_t v1_id = e_info.v1_id;
    uint32_t v2_id = e_info.v2_id;

    if (qv_id != v1_id)
        is_out_edges_shard = false;
    
    grid_off v1_label_id = _vertices_info[e_info.v1_id].vl_id;
    grid_off v2_label_id = _vertices_info[e_info.v2_id].vl_id;
    
    grid_off v1_min_gid = _cfg.gfs->get_min_grid_row(v1_label_id);
    grid_off v1_max_gid = _cfg.gfs->get_max_grid_row(v1_label_id);
    grid_off v2_min_gid = _cfg.gfs->get_min_grid_row(v2_label_id);
    grid_off v2_max_gid = _cfg.gfs->get_max_grid_row(v2_label_id);
    
    // printf("edge_idx: %u [%u] (%u->%u)\n", edge_idx, _order_rank[odr], e_info.v1_id, e_info.v2_id);
    for (uint32_t r = v1_min_gid; r < v1_max_gid; r++) {
        for (uint32_t c = v2_min_gid; c < v2_max_gid; c++) {
             // grid (r, c)
            // printf("out-shards: (%u, %u) [%u]\n", r, c, edge_idx);
            gbid_t gbid = make_gbid(r, c, e_info.el_id);
            if (!_chk_grid_range(gbid, v1_id, v2_id))
                continue;
            
            stree_cptr const stree = _cfg.gfs->stree(gbid);
            stree->post_order_dfs_without_leaf([&](stree_node_ptr xnode) {
                if (xnode->header.qnode_type == qtree_node_type::InternalNode)
                    return;

                stree_switch_node& sswch = *xnode;
                sleaf_t* sleaf_out_shard = sswch.out_edges;
                sleaf_t* sleaf_in_shard = sswch.in_edges;

                laddr_interval v1_range = sleaf_out_shard->actual_src_range();
                laddr_interval v2_range = sleaf_out_shard->actual_dst_range();
                laddr_interval grid_v1_range = sleaf_out_shard->src_range;
                laddr_interval grid_v2_range = sleaf_out_shard->dst_range;

                if (!_push_range_rslt(gbid, e_info.v1_id, e_info.v2_id, v1_range, v2_range, grid_v1_range, grid_v2_range, e_ty))
                    return;
                {   
                    if (!e_info.optional_edge_with_regular_vertex) {
                        if (is_out_edges_shard) _sleaf_rslt[edge_idx] = sleaf_out_shard;
                        else _sleaf_rslt[edge_idx] = sleaf_in_shard;
                    }
                    else {
                        if (is_out_edges_shard) _sleaf_rslt[edge_idx] = sleaf_in_shard;
                        else _sleaf_rslt[edge_idx] = sleaf_out_shard;
                    }
                    
                    sched_plan_node* node;
                    node = new sched_plan_node();
                    node->parent = parent;
                    node->level = parent->level + 1;
                    node->e_id = e_id;
                    node->v_id = qv_id + 1;
                    node->leaf_ptr = _sleaf_rslt[edge_idx];
                    node->valid = false;
                    
                    _iterate_sub_task(edge_idx + 1, node);

                    if (node->valid) {
                        if (node->parent != nullptr) {
                            (node->parent)->child.push_back(node);
                            (node->parent)->valid = true;
                        }
                    }
                    else
                        delete node;
                }
                _pop_range_rslt(e_info.v1_id, e_info.v2_id);
            });

            if (e_info.e_dir == qgraph_edge_direction::UNDIRECTED) { // if query edge is undirected, we have to check grid (c, r)
                // printf("in-shards: (%u, %u) [%u]\n", c, r, edge_idx);
                gbid_t rev_gbid = make_gbid(c, r, e_info.el_id);

                stree_cptr const rev_stree = _cfg.gfs->stree(rev_gbid);
                rev_stree->post_order_dfs_without_leaf([&](stree_node_ptr xnode) {
                    if (xnode->header.qnode_type == qtree_node_type::InternalNode)
                        return;

                    stree_switch_node& sswch = *xnode;
                    sleaf_t* sleaf_out_shard = sswch.out_edges;
                    sleaf_t* sleaf_in_shard = sswch.in_edges;

                    laddr_interval v1_range = sleaf_in_shard->actual_src_range();
                    laddr_interval v2_range = sleaf_in_shard->actual_dst_range();
                    laddr_interval grid_v1_range = sleaf_out_shard->src_range;
                    laddr_interval grid_v2_range = sleaf_out_shard->dst_range;

                    if (!_push_range_rslt(gbid, e_info.v1_id, e_info.v2_id, v1_range, v2_range, grid_v1_range, grid_v2_range, e_ty))
                        return;
                    {
                        if (!e_info.optional_edge_with_regular_vertex) {
                            if (is_out_edges_shard) _sleaf_rslt[edge_idx] = sleaf_in_shard;
                            else _sleaf_rslt[edge_idx] = sleaf_out_shard;
                        }
                        else {
                            if (is_out_edges_shard) _sleaf_rslt[edge_idx] = sleaf_out_shard;
                            else _sleaf_rslt[edge_idx] = sleaf_in_shard;
                        }
                        
                        sched_plan_node* node;
                        node = new sched_plan_node();
                        node->parent = parent;
                        node->level = parent->level + 1;
                        node->e_id = e_id;
                        node->v_id = qv_id + 1;
                        node->leaf_ptr = _sleaf_rslt[edge_idx];
                        node->valid = false;
                        
                        _iterate_sub_task(edge_idx + 1, node);

                        if (node->valid) {
                            if (node->parent != nullptr) {
                                (node->parent)->child.push_back(node);
                                (node->parent)->valid = true;
                            }
                        }
                        else 
                            delete node;
                    }
                    _pop_range_rslt(e_info.v1_id, e_info.v2_id);
                });
            }
        }
    }
}

bool lgf_scheduler::_chk_grid_range(gbid_t gbid, uint32_t src_id, uint32_t dst_id) {
    if (_range_rslt[src_id].local_range.size() != 1 && _range_rslt[src_id].grid_id != gbid.row)
        return false;
    if (_range_rslt[dst_id].local_range.size() != 1 && _range_rslt[dst_id].grid_id != gbid.col)
        return false;
    return true;
}

bool lgf_scheduler::_push_range_rslt(
    gbid_t gbid,
    uint32_t src_id, 
    uint32_t dst_id, 
    gstream::grid_format::laddr_interval src_range, 
    gstream::grid_format::laddr_interval dst_range,
    gstream::grid_format::laddr_interval max_src_range, 
    gstream::grid_format::laddr_interval max_dst_range,
    qgraph_edge_type e_type) {
    using namespace gstream;
    using namespace grid_format;
    using namespace detail;

    // replace ranges
    if (e_type == qgraph_edge_type::OPTIONAL) {
        if (_order_rank[src_id] < _order_rank[dst_id]) {
            if (!is_overlap(_range_rslt[src_id].local_range.top(), max_src_range))
                return false;
        }
        else {
            if (!is_overlap(_range_rslt[dst_id].local_range.top(), max_dst_range))
                return false;
        }
    }
    else if (e_type == qgraph_edge_type::NEGATIVE) {
        if (!is_overlap(_range_rslt[src_id].local_range.top(), max_src_range))
            return false;
        if (!is_overlap(_range_rslt[dst_id].local_range.top(), max_dst_range))
            return false;
    }
    else {
        if (!is_overlap(_range_rslt[src_id].local_range.top(), src_range))
            return false;
        if (!is_overlap(_range_rslt[dst_id].local_range.top(), dst_range))
            return false;
    }

    if (_range_rslt[src_id].local_range.size() == 1)
        _range_rslt[src_id].grid_id = gbid.row;
    if (_range_rslt[dst_id].local_range.size() == 1)
        _range_rslt[dst_id].grid_id = gbid.col;

    assert(src_id != dst_id);
    _range_rslt[src_id].local_range.push({
            std::max(_range_rslt[src_id].local_range.top().lo, src_range.lo),
            std::min(_range_rslt[src_id].local_range.top().hi, src_range.hi)
        });
    _range_rslt[dst_id].local_range.push({
            std::max(_range_rslt[dst_id].local_range.top().lo, dst_range.lo),
            std::min(_range_rslt[dst_id].local_range.top().hi, dst_range.hi)
        });

    return true;
}

void lgf_scheduler::_pop_range_rslt(uint32_t src_id, uint32_t dst_id) {
    _range_rslt[src_id].local_range.pop();
    _range_rslt[dst_id].local_range.pop();
}

void lgf_scheduler::_print_scheduler() {
    printf("> qgraph_adj:\n");
    for (uint32_t i = 0; i < _num_vertices; i++) {
        printf("\tadj[%u]: ", i);
        for(uint32_t j = 0; j < _adj_list[i].size(); j++) {
            qgraph_edge_info& e_info = _edges_info[_adj_list[i][j].edge_id];
            printf("(%u, %u) ", e_info.v1_id, e_info.v2_id);
        }
        printf("\n");
    }

    printf("> scheduling plan:\n");
    _sched_plan_tree.print_sched_plan();
}

} // namespace cu_match
