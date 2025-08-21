#ifndef GSTREAM_GRID_FORMAT_DETAIL_EL32_OFSTREAM_H
#define GSTREAM_GRID_FORMAT_DETAIL_EL32_OFSTREAM_H
#include <string>
#include <fstream>

namespace gstream {
namespace grid_format {
namespace detail {
namespace gridgen {

class el32_ofstream {
public:
    el32_ofstream();
    ~el32_ofstream();

    void init_stream(std::string const& output_file_path);
    void write_stream(uint32_t const& row_v_ID, uint32_t const& col_v_ID);
    void close_stream();
    void init_buf(uint64_t const& output_buf_size);

private:
    std::string _output_path;
    std::ofstream _ofs;
    uint32_t* _out_buf;
    uint64_t _output_buf_size;
    uint64_t _offset;

    void _clear_buf();
    void _flush_buf();
};

} // namespace gridgen
} // namespace detail
} // namespace grid_format
} // namespace gstream
#endif // GSTREAM_GRID_FORMAT_DETAIL_EL32_OFSTREAM_H
