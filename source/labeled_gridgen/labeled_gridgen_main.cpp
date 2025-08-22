#include <gstream/grid_format/labeled_gridgen.h>
#include <ash/utility/dbg_log.h>
#include <ash/stop_watch.h>
#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>

using namespace gstream;
using namespace grid_format::detail::gridgen;

using program_config = labeled_grid_generator::config_t;

constexpr bool DEFAULT_HAS_HEADER = true;
constexpr bool DEFAULT_SKIP_RELABELING = false;

program_config make_config(int argc, char* argv[]) {
    program_config cfg;
    memset(&cfg, 0, sizeof(program_config));

    cfg.grid_name = argv[1];
    cfg.indir = argv[2];
    cfg.outdir = argv[3];
    cfg.schema_path = argv[4];

    if (argc > 5) {
        if (std::tolower(argv[5][0]) == 'y')
            cfg.has_header = true;
        else if (std::tolower(argv[5][0]) == 'n')
            cfg.has_header = false;
        else
            ASH_FATAL("Parameter $5: Has header flag is wrong");
    }
    else {
        cfg.has_header = DEFAULT_HAS_HEADER;
    }

    if (argc > 6) {
        if (std::tolower(argv[6][0]) == 'y')
            cfg.skip_relabeling = true;
        else if (std::tolower(argv[6][0]) == 'n')
            cfg.skip_relabeling = false;
        else
            ASH_FATAL("Parameter $6: Skip relabeling flag is wrong");
    } else {
        cfg.skip_relabeling = DEFAULT_SKIP_RELABELING;
    }

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
    std::cout << std::left << std::setw(column_width) << "Input directory:"
        << "| " << std::left << std::setw(column_width) << cfg.indir << "\n";
    std::cout << std::left << std::setw(column_width) << "Output directory:"
        << "| " << std::left << std::setw(column_width) << cfg.outdir << "\n";
    std::cout << std::left << std::setw(column_width) << "Schema file path:"
        << "| " << std::left << std::setw(column_width) << cfg.schema_path << "\n";
    std::cout << std::left << std::setw(column_width) << "Header flag:"
        << "| " << std::left << std::setw(column_width) << std::boolalpha << cfg.has_header << "\n";
    std::cout << std::left << std::setw(column_width) << "Skip relabeling:"
        << "| " << std::left << std::setw(column_width) << std::boolalpha << cfg.skip_relabeling << "\n";
    std::cout << std::string(column_width, '-') << "+" << std::string(column_width, '-') << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("usage> %s <$1> <$2> <$3> <$4> <$5> <$6>\n"
            "$1: Graph name: [string] Output graph name\n"
            "$2: Input directory: [string] Input file directory\n"
            "$3: Output directory: [string] Output file directory, Note that program does not make new directory!\n"
            "$4: Schema file path: [string] Schema file path, File extension must be json!\n"
            "$5: Has header flag: [y/n] Whether to include CSV headers\n"
            "$6: Skip relabeling flag: [y/n] Whether to skip relabeling vertex process\n",
            argv[0]);
        return -1;
    }

    program_config const prog_cfg = make_config(argc, argv);
    print_args(prog_cfg);

    ash::stop_watch watch;

    {
        labeled_grid_generator::config_t labeled_gridgen_cfg = prog_cfg;
        labeled_grid_generator gridgen;
        gridgen.exec(labeled_gridgen_cfg);
    }

    watch.lab();
    std::cout << "Elapsed time: " << watch.elapsed_sec() << " seconds\n";

    return 0;
}
