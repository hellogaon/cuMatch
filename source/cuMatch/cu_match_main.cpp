#include <cu_match/subgraph_matcher.h>
#include <ash/utility/dbg_log.h>
#include <ash/numeric.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>

using namespace cu_match;

using program_config = subgraph_matcher::config_t;

program_config make_config(char* argv[]) {
    program_config cfg;
    memset(&cfg, 0, sizeof(program_config));

    cfg.grid_name = argv[1];
    cfg.grid_dir = argv[2];
    cfg.query_file_path = argv[3];
    cfg.host_buffer_size = ash::GiB(strtoull(argv[4], nullptr, 10));
    cfg.device_buffer_size = ash::GiB(strtoull(argv[5], nullptr, 10));
    if (std::tolower(argv[6][0]) == 'y')
        cfg.is_table_join = true;
    else if (std::tolower(argv[6][0]) == 'n')
        cfg.is_table_join = false;
    else
        ASH_FATAL("Parameter $6: Table join flag is wrong");
    if (std::tolower(argv[7][0]) == 'y')
        cfg.in_memory_mode = true;
    else if (std::tolower(argv[7][0]) == 'n')
        cfg.in_memory_mode = false;
    else
        ASH_FATAL("Parameter $7: in memory mode flag is wrong");
    if (std::tolower(argv[8][0]) == 'y')
        cfg.gpu_streaming_mode = true;
    else if (std::tolower(argv[8][0]) == 'n')
        cfg.gpu_streaming_mode = false;
    else
        ASH_FATAL("Parameter $8: GPU streaming mode flag is wrong");
    

    return cfg;
}

void print_args(program_config const& cfg) {
    constexpr uint32_t column_width = 25;
    std::cout << std::string(column_width, '-') << "+" << std::string(column_width, '-') << "\n";
    std::cout << std::left << std::setw(column_width) << "Argument"
        << "|" << std::left << std::setw(column_width) << "Value" << "\n";
    std::cout << std::string(column_width, '-') << "+" << std::string(column_width, '-') << "\n";
    std::cout << std::left << std::setw(column_width) << "Grid name:"
        << "| " << std::left << std::setw(column_width) << cfg.grid_name << "\n";
    std::cout << std::left << std::setw(column_width) << "Grid directory:"
        << "| " << std::left << std::setw(column_width) << cfg.grid_dir << "\n";
    std::cout << std::left << std::setw(column_width) << "Query file path:"
        << "| " << std::left << std::setw(column_width) << cfg.query_file_path << "\n";
    std::cout << std::left << std::setw(column_width) << "Host buffer size:"
        << "| " << std::left << std::setw(column_width) << cfg.host_buffer_size << "\n";
    std::cout << std::left << std::setw(column_width) << "Device buffer size:"
        << "| " << std::left << std::setw(column_width) << cfg.device_buffer_size << "\n";
    std::cout << std::left << std::setw(column_width) << "Table join flag:"
        << "| " << std::left << std::setw(column_width) << std::boolalpha << cfg.is_table_join << "\n";
    std::cout << std::left << std::setw(column_width) << "In-memory mode:"
        << "| " << std::left << std::setw(column_width) << std::boolalpha << cfg.in_memory_mode << "\n";
    std::cout << std::left << std::setw(column_width) << "GPU streaming mode:"
        << "| " << std::left << std::setw(column_width) << std::boolalpha << cfg.gpu_streaming_mode << "\n";
    std::cout << std::string(column_width, '-') << "+" << std::string(column_width, '-') << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        printf("usage> %s <$1> <$2> <$3> <$4> <$5> <$6> <$7> <$8>\n"
            "$1: Graph name: [string]\n"
            "$2: Grid directory: [string]\n"
            "$3: Query file path: [string]\n"
            "$4: Host buffer size: [integer]\n"
            "$5: Device buffer size: [integer]\n"
            "$6: Table join flag: [y/n]\n"
            "$7: In-memory mode flag: [y/n]\n"
            "$8: GPU streaming mode flag: [y/n]\n",
            argv[0]);
        return -1;
    }

    program_config const prog_cfg = make_config(argv);
    print_args(prog_cfg);

    {
        subgraph_matcher::config_t sm_cfg = prog_cfg;
        subgraph_matcher matcher;
        matcher.exec(sm_cfg);
    }

    return 0;
}
