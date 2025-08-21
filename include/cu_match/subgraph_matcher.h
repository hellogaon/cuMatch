#ifndef CU_MATCH_SUBGRAPH_MATCHER_H
#define CU_MATCH_SUBGRAPH_MATCHER_H
#include <cu_match/simple_parser.h>
#include <cu_match/query_graph.h>
#include <cu_match/matching_order_builder.h>
#include <cu_match/lgf_scheduler.h>
#include <cu_match/device_manager.cuh>
#include <gstream/grid_format/grid_file_stream.h>
#include <string>

namespace cu_match {

class subgraph_matcher {

public:
    struct config_t {
        char const* grid_name;
        char const* grid_dir;
        char const* query_file_path;
        uint64_t host_buffer_size;
        uint64_t device_buffer_size;
        bool is_table_join;
        bool in_memory_mode;
        bool gpu_streaming_mode;
    };
    
    subgraph_matcher();
    ~subgraph_matcher() noexcept;
    void exec(config_t const& cfg); 

private:
    config_t                    _cfg;
    simple_parser*              _parser;
    gstream::grid_file_stream*  _gfs;
    query_graph*                _qgraph;
    matching_order_builder*     _order_builder;
    lgf_scheduler*              _sched;
    device_manager*             _dev_mgr;
    
    uint32_t                    _num_query_vertices;
    uint32_t                    _num_query_edges;
    uint32_t                    _num_constraints;

    bool _init(config_t const& cfg);
    void _cleanup();
};

} // namespace cu_match

#endif // CU_MATCH_SUBGRAPH_MATCHER_H
