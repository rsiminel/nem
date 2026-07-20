#include <iostream>
#include <stdexcept>

#include "cli.hpp"

int main(int argc, char** argv) {
    try {
        nem::cli::Options opts = nem::cli::parse_args(argc, argv);
        return nem::cli::run_dispatch(opts);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
