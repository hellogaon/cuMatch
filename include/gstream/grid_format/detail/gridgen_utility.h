#ifndef GSTREAM_GRID_FORMAT_DETAIL_GRIDGEN_UTILITY_H
#define GSTREAM_GRID_FORMAT_DETAIL_GRIDGEN_UTILITY_H
#include <gstream/grid_format/grid_format_defines.h>
#include <string>
#include <vector>

namespace gstream {
namespace grid_format {
namespace detail {

gbid_t gbid_from_path(std::string const& ph, char delim);
gbid_t gbid3_from_path(std::string const& ph, char delim);
std::string make_input_path(char const* indir, char const* format, gbid_t gbid, char const* ext);
std::string make_grid_format_path(char const* outdir, char const* filename_format, ...);
uint32_t find_max_gb_index(char const* dir, char const* ext);
uint32_t find_max_gb_index(std::string const path_arr[], size_t n);
std::vector<std::string> make_input_file_list(char const* dir, char const* ext);

} // namespace detail

template <typename Range>
Range divide_range(Range const range, char const indicator) {
    if (range.hi == 0) {
        printf("DBG: input range is zero { %u, %u }\n", range.lo, range.hi);
        assert(false);
    }
    assert(range.hi > 0);
    typename Range::value_type length = range.hi - range.lo + 1;

    /*if (!ash::is_power_of_two(length)) {
        printf("DBG: length is not power of two. length = %u from { %u, %u }\n", length, range.lo, range.hi);
        assert(false);
    }*/
    assert(ash::is_power_of_two(length));

    Range r;
    if (indicator == 0) {
        //r = Range{ range.lo, range.hi / 2 - 1 };
        r = Range{ range.lo, range.hi - length / 2 };
    }
    else {
        r = Range{ range.lo + length / 2, range.hi };
    }

    /*if (r.lo > r.hi) {
        printf("DBG: [r.lo > r.hi] { %u, %u }\n", r.lo, r.hi);
        assert(false);
    }*/
    assert(r.lo <= r.hi);

    /*if (!ash::is_power_of_two(r.width())) {
        printf("DBG: Width is not power of two. width: %u from { %u, %u } / Input: { %u, %u, %u }\n", r.width(), r.lo, r.hi, range.lo, range.hi, indicator);
        assert(false);
    }*/
    assert(ash::is_power_of_two(r.width()));

    return r;
}

inline bool is_overlap(laddr_interval const& range1, laddr_interval const& range2) {
    // assert(range1.lo <= range1.hi);
    // assert(range2.lo <= range2.hi);

    //return (range1.lo <= range2.hi) && (range2.lo <= range1.hi);
    return std::max(range1.lo, range2.lo) <= std::min(range1.hi, range2.hi);
}

inline bool is_overlap(gaddr_interval const& range1, gaddr_interval const& range2) {
    // assert(range1.lo <= range1.hi);
    // assert(range2.lo <= range2.hi);

    //return (range1.lo <= range2.hi) && (range2.lo <= range1.hi);
    return std::max(range1.lo, range2.lo) <= std::min(range1.hi, range2.hi);
}

} // namespace grid_format
} // namespace gstream

#endif // GSTREAM_GRID_FORMAT_DETAIL_GRIDGEN_UTILITY_H
