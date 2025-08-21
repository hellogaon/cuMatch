#include "csv_ofstream.h"
#include <cassert>
#include <cstring>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

csv_ofstream::csv_ofstream() {
}

csv_ofstream::~csv_ofstream() {
}

void csv_ofstream::init_stream(std::string const& output_file_path) {
    _output_path = output_file_path;
    _ofs.open(output_file_path, std::ios::out | std::ios::binary);
    assert(_ofs.is_open());
}

void csv_ofstream::write_header(uint32_t const& num_cols) {
    if (num_cols == 1) {
        std::string out = ":ID\n";
        _ofs.write(out.c_str(), out.size());
    }
    else if (num_cols == 2) {
        std::string out = ":START_ID|:END_ID\n";
        _ofs.write(out.c_str(), out.size());
    }
}

void csv_ofstream::write_single_integer(uint64_t const& ID) {
    std::string out = std::to_string(ID) + "\n";
    _ofs.write(out.c_str(), out.size());
}

void csv_ofstream::write_double_integer(uint64_t const& ID1, uint64_t const& ID2) {
    std::string out = std::to_string(ID1) + "|" + std::to_string(ID2) + "\n";
    _ofs.write(out.c_str(), out.size());
}

void csv_ofstream::close_stream() {
    namespace fs = ::boost::filesystem;
    _ofs.close();
    if (!fs::file_size(_output_path))
        fs::remove(_output_path);
    _ofs.clear();
}

void csv_ofstream::init_buf() {

}

void csv_ofstream::_clear_buf() {
    
}
