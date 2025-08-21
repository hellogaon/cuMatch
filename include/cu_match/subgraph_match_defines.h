#ifndef CU_MATCH_SUBGRAPH_MATCH_DEFINES_H
#define CU_MATCH_SUBGRAPH_MATCH_DEFINES_H
#include <gstream/grid_format/grid_format_defines.h>
#include <gstream/grid_format/grid_file_stream.h>
#include <stdlib.h>

namespace cu_match {

constexpr unsigned NameSizeLimit = 32;
constexpr unsigned LabelSizeLimit = gstream::grid_format::LabelSizeLimit;
constexpr unsigned MaxNumQueryVertices = 8;
constexpr unsigned MaxNumTreeNodes = 500;
constexpr unsigned MaxNumShardPerIter = 300;
constexpr unsigned MaxNumQueryEdges = 10;
constexpr unsigned MaxNumConstraints = 5;
constexpr unsigned MaxNumQueryDegrees = 4;
constexpr unsigned MaxNumVerticesPerGrid = (1 << 24);
constexpr unsigned NULL_VALUE = MaxNumVerticesPerGrid;

using gbid_t = gstream::gbid_t;
using sleaf_ptr = gstream::sleaf_ptr;
using order_element = uint32_t;
using flip24_shard = gstream::grid_format::flip24_shard;
using flip24_shard_ptr = flip24_shard*;

enum class qgraph_edge_type : char {
    OPTIONAL = 0,
    NEGATIVE = 1,
    REGULAR = 2
};

enum class qgraph_edge_direction : char {
    DIRECTED = 0,
    UNDIRECTED = 1
};

enum class qgraph_constraint_type : char {
    NOT_EQUAL = 0,
    REGULAR = 1
};

// for scheduling plan
template <typename NodePointer>
struct sched_plan_node_info {
    NodePointer                parent;    // 8-byte
    uint32_t                   level;     // 4-bite
    uint32_t                   e_id;      // 4-bite
    uint32_t                   v_id;      // 4-bite (group with same v_id)
    sleaf_ptr                  leaf_ptr;  // 8-byte
    std::vector<NodePointer>   child;     // 8-byte
    bool                       valid;     // 1-byte
};

// for query graph
struct qgraph_vertex_info {
    gstream::grid_off vl_id;
    char name[NameSizeLimit];
    uint32_t degree;
    bool is_optional;
};

struct qgraph_edge_info {
    gstream::grid_off el_id;
    uint32_t v1_id;
    uint32_t v2_id;
    qgraph_edge_type e_type;
    qgraph_edge_direction e_dir;
    bool optional_edge_with_regular_vertex;
};

struct qgraph_constraint_info {
    uint32_t v1_id;
    uint32_t v2_id;
    qgraph_constraint_type c_type;
};

// for parser
struct parse_vertex_info {
    char label[LabelSizeLimit];
    char name[NameSizeLimit];
};

struct parse_edge_info {
    char label[LabelSizeLimit];
    char v1_name[NameSizeLimit];
    char v2_name[NameSizeLimit];
    qgraph_edge_type e_type;
    qgraph_edge_direction e_dir;
};
struct parse_constraint_info {
    char v1_name[NameSizeLimit];
    char v2_name[NameSizeLimit];
    qgraph_constraint_type c_type;
};

} // namespace cu_match

#endif // CU_MATCH_QUERY_GRAPH_H
