#ifndef GSTREAM_GRID_FORMAT_DETAIL_LABELED_GRIDGEN_UTILITY_H
#define GSTREAM_GRID_FORMAT_DETAIL_LABELED_GRIDGEN_UTILITY_H
#include <gstream/grid_format/grid_format_defines.h>
#include <string>
#include <vector>

namespace gstream {
namespace grid_format {
namespace detail {

std::vector<std::string> make_input_file_list(char const* dir, char const* ext);
bool el32_check_output_directory(std::string const& path, uint32_t row);
std::string el32_make_path(std::string const& path, uint32_t const& row_grid_id, uint32_t const& col_grid_id, uint32_t const& edge_label_id);
std::string el32_make_temporary_path(std::string const& path);
std::string el32_make_subdir_path(std::string const& path, uint32_t const& row_grid_id);
void el32_remove_temorary_path(std::string const& path);
std::string make_labeled_input_path(char const* indir, std::string& sub_path);

} // namespace detail
} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_DETAIL_LABELED_GRIDGEN_UTILITY_H
