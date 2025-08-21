#include <string>
#include <fstream>

class csv_ofstream {
public:
    csv_ofstream();
    ~csv_ofstream();

    void init_stream(std::string const& output_file_path);
    void write_header(uint32_t const& num_cols);
    void write_single_integer(uint64_t const& ID);
    void write_double_integer(uint64_t const& ID1, uint64_t const& ID2);
    void close_stream();
    void init_buf();

private:
    std::string _output_path;
    std::ofstream _ofs;

    void _clear_buf();
};
