#ifndef GSTREAM_GRID_FORMAT_DEGREE_BLOCK_BUILDER_H
#define GSTREAM_GRID_FORMAT_DEGREE_BLOCK_BUILDER_H
#include <gstream/grid_format/grid_format_defines.h>

namespace gstream {
namespace grid_format {
namespace detail {
namespace gridgen {

class degree_block_builder {
public:
    struct config_t {
        grid_dim dim;
    };
    struct degblk_index_t {
        degree_type_code unit_type;
        void* degblk;
    };
    degree_block_builder();
    ~degree_block_builder();
    bool init(config_t cfg);
    void aggregate_local_degree(degree_type type, grid_off ind, uint32_t* deg_loc) const;

    degblk_index_t const& indeg_index(grid_off const ind) const {
        return _buffer.indeg[ind];
    }

    degblk_index_t const& outdeg_index(grid_off const ind) const {
        return _buffer.outdeg[ind];
    }

private:
    config_t _cfg;
    struct {
        uint64_t* agg;
        degblk_index_t* indeg;
        degblk_index_t* outdeg;
    } _buffer;
};


} // namespace gridgen
} // namespace detail
} // namespace grid_format
} // namespace gstream
#endif // GSTREAM_GRID_FORMAT_DEGREE_BLOCK_BUILDER_H
