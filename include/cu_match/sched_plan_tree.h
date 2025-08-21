#ifndef CU_MATCH_SCHED_PLAN_TREE_H
#define CU_MATCH_SCHED_PLAN_TREE_H
#include <cu_match/subgraph_match_defines.h>
#include <cu_match/device_defines.cuh>
#include <vector>
#include <unordered_set>

namespace cu_match {

struct sched_plan_node;
using sched_plan_node_ptr = sched_plan_node*;

struct sched_plan_node: sched_plan_node_info<sched_plan_node_ptr> {
};

class sched_plan_tree {

public:
    struct config_t {
        sched_plan_node* root_node;
        uint32_t max_level;
        uint32_t num_query_vertices;
        uint32_t num_query_edges;
        uint32_t num_optional_vertices;
        std::vector<uint32_t>* postfix_order;
        std::vector<uint32_t>* optional_postfix_order;
        std::vector<uint32_t>* elevel_to_vlevel;
        bool is_table_join;
    };

    sched_plan_tree();
    ~sched_plan_tree() noexcept;
    void init(config_t const& cfg);
    void pre_order_dfs(std::function<void(sched_plan_node*)> const& visit);
    void post_order_dfs(std::function<void(sched_plan_node*)> const& visit);
    bool pop_grid_group();
    uint64_t pop_single_kernel_task(single_kernel_task_info* k_info, uint64_t capacity, bool initial);
    void print_sched_plan();

private:
    struct tracking_info {
        sched_plan_node* node;
        uint32_t child_idx;
    };
    struct grid_group_info {
        uint32_t v_id;
        uint32_t level;
        uint32_t min_level;
        uint32_t max_level;
        std::vector<sleaf_ptr> grid_group;
        subtask_info* st_info;
    };
    struct tree_info {
        uint32_t v_level;
        uint32_t subtask_id;
    };

    config_t _cfg;
    sched_plan_node* _root_node;
    grid_group_info _tmp_group_info;
    grid_group_info* _grid_group_ptr;
    std::vector<tracking_info> _tracking_stk;
    std::unordered_set<gstream::grid_format::shard_uid> _cache;
    std::vector<tree_info> _tree_info_vec;

    void _cleanup();
    void _iterate_init();
    void _backtracking(sched_plan_node* now, std::vector<sched_plan_node*>& rslt);
    sched_plan_node* _this_node();
    sched_plan_node* _next_node(); // pre-order scan without recursive
    subtask_info* make_subtask_info();
    void _regenerate_LFTJ_grid_group_tree(single_kernel_task_info* k_info); // for attribute join (LFTJ)
    void _regenerate_MJ_grid_group_tree(single_kernel_task_info* k_info); // for table join
    uint32_t _convert_elevel_to_vlevel(uint32_t level);
};

} // namespace cu_match

#endif // CU_MATCH_SCHED_PLAN_TREE_H
