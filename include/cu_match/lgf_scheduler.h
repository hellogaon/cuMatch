#ifndef CU_MATCH_LGF_SCHEDULER_H
#define CU_MATCH_LGF_SCHEDULER_H
#include <cu_match/subgraph_match_defines.h>
#include <cu_match/device_defines.cuh>
#include <cu_match/query_graph.h>
#include <cu_match/sched_plan_tree.h>
#include <gstream/grid_format/grid_file_stream.h>
#include <stack>

namespace cu_match {

class lgf_scheduler {

public:
    struct config_t {
        query_graph* qgraph;
        bool is_table_join;
        gstream::grid_file_stream* gfs;
    };
    
    lgf_scheduler();
    ~lgf_scheduler() noexcept;
    bool make_LGF_schedule(order_element* order, order_element* edge_order); 
    void init(config_t const& cfg);
    void init_global_kernel_info(global_kernel_task_info* gk_info);

    sched_plan_tree* get_sched_plan_tree() {
        return &_sched_plan_tree;
    }

private:
    config_t _cfg;
    uint32_t _num_vertices;
    uint32_t _num_edges;
    uint32_t _num_constraints;
    uint32_t _num_optional_vertices;

    struct _edge_order_info {
        uint32_t edge_id;
        uint32_t order;
    };

    struct _grid_range_info {
        gstream::grid_off grid_id;
        std::stack<gstream::grid_format::laddr_interval> local_range;
    };

    struct _adj_list_info {
        uint32_t edge_id;
        uint32_t opposite_order_id;
    };

    qgraph_vertex_info*                          _vertices_info;
    qgraph_edge_info*                            _edges_info;
    qgraph_constraint_info*                      _constraint_info;
    order_element*                               _order; // order by (regular intersection vertices) + (regular postfix_order_vertices) + (post intersection vertices) + (post postfix_order_vertices)
    std::vector<uint32_t>                        _order_rank;
    std::vector<uint32_t>                        _postfix_order;
    std::vector<uint32_t>                        _optional_postfix_order;
    std::vector<std::vector<_adj_list_info> >    _adj_list;
    std::vector<_edge_order_info>                _edge_order;
    std::vector<uint32_t>                        _elevel_to_vlevel; // convert edge level to vertex level

    // intermediate data
    std::vector<sleaf_ptr>                 _sleaf_rslt;
    std::vector<_grid_range_info>          _range_rslt;

    sched_plan_tree                        _sched_plan_tree;

    void _cleanup();
    bool _make_LFTJ_schedule(order_element* order, order_element* edge_order); 
    bool _make_MJ_schedule(order_element* order, order_element* edge_order); 
    sched_plan_node* _make_sched_plan();
    void _iterate_sub_task(uint32_t eo_id, sched_plan_node* parent);
    bool _chk_grid_range(gbid_t gbid, uint32_t src_id, uint32_t dst_id);
    bool _push_range_rslt(gbid_t gbid, uint32_t src_id, uint32_t dst_id, gstream::grid_format::laddr_interval src_range, gstream::grid_format::laddr_interval dst_range, gstream::grid_format::laddr_interval max_src_range, gstream::grid_format::laddr_interval max_dst_range, qgraph_edge_type e_type);
    void _pop_range_rslt(uint32_t src_id, uint32_t dst_id);
    void _print_scheduler();
};

} // namespace cu_match

#endif // CU_MATCH_LGF_SCHEDULER_H
