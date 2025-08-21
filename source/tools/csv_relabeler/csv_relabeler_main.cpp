// Real-world graph relabeler (simple LSQB version)
#include "program_internals.h"
#include <gstream/grid_format/grid_file_stream.h>
#include <ash/utility/dbg_log.h>
#include <ash/stop_watch.h>
#include <iostream>
#include <iomanip>

struct program_args {
    char const* input_path;
    char const* output_path;
    char const* schema_path;
    bool has_header;
};

program_args make_args(char const* argv[]) {
    program_args result;
    result.input_path = argv[1];
    result.output_path = argv[2];
    result.schema_path = argv[3];
    if (std::tolower(argv[4][0]) == 'y')
        result.has_header = true;
    else if (std::tolower(argv[4][0]) == 'n')
        result.has_header = false;
    else
        ASH_FATAL("Parameter $4: has_header flag is wrong");
    return result;
}

int main(int argc, char const* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <input_path> <output_path> <schema path> <has_header>\n";
        return 1;
    }
    program_args const prog_args = make_args(argv);

    ash::stop_watch watch;
    {
        csv_relabeler converter;
        csv_relabeler::config_t converter_cfg;
        converter_cfg.input_path = prog_args.input_path;
        converter_cfg.output_path = prog_args.output_path;
        converter_cfg.schema_path = prog_args.schema_path;
        converter_cfg.has_header = prog_args.has_header;
        if (!converter.run(converter_cfg)) {
            ASH_ERRLOG("CSV preprocessor returns error!");
            return 2;
        }
    }
    watch.lab();
    std::cout << "Elapsed time: " << watch.elapsed_sec() << " seconds\n";

    return 0;
}
