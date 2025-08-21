#ifndef GSTREAM_GRID_FORMAT_GRID_FILE_STREAM_DEFINES_H
#define GSTREAM_GRID_FORMAT_GRID_FILE_STREAM_DEFINES_H
#include <stdint.h>

namespace gstream {
namespace grid_format {

class grid_file_stream;

struct extent_load_info;
struct extent_config;
struct extent_t;

using extent_id_t = uint64_t;
constexpr static extent_id_t InvalidExtentID = UINT64_MAX;

class  shard_tree;
struct stree_internal_node;
struct stree_switch_node; // for LGF
struct stree_leaf_node;
struct stree_node_block;

using stree_t = shard_tree;
using stree_ptr = shard_tree*;
using stree_cptr = shard_tree const*;
using sintn_t = stree_internal_node;
using sswch_t = stree_switch_node; // for LGF
using sleaf_t = stree_leaf_node;
using snode_t = stree_node_block;
using sleaf_ptr = sleaf_t*;
using sleaf_cptr = sleaf_t const*;
using readonly_sleaf_ptrarr = sleaf_cptr const*;
using mutable_sleaf_ptrarr = sleaf_cptr*;

struct stree_node_header;

namespace detail {

class extent_mapper;
class extent_map_generator;

} // ns detail

} // ns grid_format

using grid_format::grid_file_stream;
using grid_format::shard_tree;
using grid_format::stree_t;
using grid_format::stree_ptr;
using grid_format::stree_cptr;
using grid_format::stree_internal_node;
using grid_format::sintn_t;
using grid_format::stree_leaf_node;
using grid_format::sleaf_t;
using grid_format::sleaf_ptr;
using grid_format::sleaf_cptr;
using grid_format::readonly_sleaf_ptrarr;
using grid_format::mutable_sleaf_ptrarr;

} // ns gstream

#endif // GSTREAM_GRID_FORMAT_GRID_FILE_STREAM_DEFINES_H
