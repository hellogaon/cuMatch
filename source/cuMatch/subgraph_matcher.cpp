#include <cu_match/subgraph_matcher.h>
#include <ash/utility/dbg_log.h>
#include <ash/stop_watch.h>
#include <iostream>

namespace cu_match {

subgraph_matcher::subgraph_matcher() {}

subgraph_matcher::~subgraph_matcher() {
    _cleanup();
}

void subgraph_matcher::exec(config_t const& cfg) {
    if (!_init(cfg))
        return;

    ash::stop_watch watch;

    // generate matching order
    if (!_order_builder->gen_matching_order()) {
        ASH_ERRLOG("Failed to generate matching order");
        return;
    }
    // make LGF schedule
    if (!_sched->make_LGF_schedule(_order_builder->order(), _order_builder->edge_order())) {
        ASH_ERRLOG("Failed to make LGF schedule");
        return;
    }

    watch.lab();
    std::cout << "> Scheduling elapsed time: " << watch.elapsed_sec() << " seconds\n";
    watch.reset();

    // run subgraph matching
    if (!_cfg.gpu_streaming_mode) {
        if (!_dev_mgr->execute_match(_sched)) {
            ASH_ERRLOG("Failed to execute kernel");
            return;
        }
    }
    else {
        if (!_dev_mgr->execute_match_with_double_buffering(_sched)) {
            ASH_ERRLOG("Failed to execute kernel");
            return;
        }
    }

    watch.lab();
    std::cout << "> Query elapsed time: " << watch.elapsed_sec() << " seconds\n";
}

bool subgraph_matcher::_init(config_t const& cfg) {
    using namespace gstream::grid_format;
    _cfg = cfg;

    if (_cfg.host_buffer_size < _cfg.device_buffer_size) {
        printf("host_buffer_size must be sufficiently larger than device_buffer_size!\n");
        exit(1);
    }

    // open grid graph
    {
        grid_file_stream::config_t gfs_cfg;
        gfs_cfg.grid_dir = _cfg.grid_dir;
        gfs_cfg.grid_name = _cfg.grid_name;
        gfs_cfg.extent_size = 0;
        gfs_cfg.stream_opt = GFS_USE_DIRECT_IO;
        _gfs = new gstream::grid_file_stream;
        if (!_gfs->open(gfs_cfg)) {
            ASH_ERRLOG("Failed to open grid_file_stream");
            return false;
        }
    }
    // parse query
    {
        simple_parser::config_t parser_cfg;
        parser_cfg.query_file_path = _cfg.query_file_path;
        _parser = new simple_parser;
        if (!_parser->parse(parser_cfg)) {
            ASH_ERRLOG("Failed to parse query");
            return false;
        }
        _num_query_vertices = _parser->num_query_vertices();
        _num_query_edges = _parser->num_query_edges();
        _num_constraints = _parser->num_constraints();
    }
    // make query graph    
    {
        query_graph::config_t qgraph_cfg;
        qgraph_cfg.num_vertices = _num_query_vertices;
        qgraph_cfg.num_edges = _num_query_edges;
        qgraph_cfg.num_constraints = _num_constraints;
        qgraph_cfg.parse_vertices_info = _parser->parse_vertices_info();
        qgraph_cfg.parse_edges_info = _parser->parse_edges_info();
        qgraph_cfg.parse_constraints_info = _parser->parse_constraints_info();
        qgraph_cfg.gfs = _gfs;
        _qgraph = new query_graph;
        if (!_qgraph->make_qgraph(qgraph_cfg)) {
            ASH_ERRLOG("Failed to make query graph");
            return false;
        }
    }
    // matching order init
    {
        matching_order_builder::config_t order_cfg;
        order_cfg.qgraph = _qgraph;
        order_cfg.is_table_join = _cfg.is_table_join;
        _order_builder = new matching_order_builder;
        _order_builder->init(order_cfg);
    }
    // LGF scheduler init
    {
        lgf_scheduler::config_t sched_cfg;
        sched_cfg.qgraph = _qgraph;
        sched_cfg.is_table_join = _cfg.is_table_join;
        sched_cfg.gfs = _gfs;
        _sched = new lgf_scheduler;
        _sched->init(sched_cfg);
    }
    // kernel executor init
    {
        device_manager::config_t devmgr_cfg;
        devmgr_cfg.qgraph = _qgraph;
        devmgr_cfg.host_buffer_size = _cfg.host_buffer_size;
        devmgr_cfg.device_buffer_size = _cfg.device_buffer_size;
        devmgr_cfg.gfs = _gfs;
        devmgr_cfg.is_table_join = _cfg.is_table_join;
        devmgr_cfg.in_memory_mode = _cfg.in_memory_mode;
        _dev_mgr = new device_manager;
        _dev_mgr->init(devmgr_cfg);
    }
    return true;
}

void subgraph_matcher::_cleanup() {
    assert(_dev_mgr != nullptr);
    delete _dev_mgr;
    _dev_mgr = nullptr;

    assert(_sched != nullptr);
    delete _sched;
    _sched = nullptr;

    assert(_order_builder != nullptr);
    delete _order_builder;
    _order_builder = nullptr;
    
    assert(_qgraph != nullptr);
    delete _qgraph;
    _qgraph = nullptr;

    assert(_parser != nullptr);
    delete _parser;
    _parser = nullptr;

    assert(_gfs != nullptr);
    delete _gfs;
    _gfs = nullptr;
}

} // namespace cu_match
