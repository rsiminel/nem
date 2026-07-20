#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fastnem/io.hpp>
#include <fastnem/ppanggolin.hpp>

namespace nem::cli {

    struct Options {
        std::string feature_path;
        std::string graph_path;
        std::string out_dir = ".";
        std::size_t k = 3;
        double beta = 2.5;
        std::size_t sm_degree = 10;
        bool free_dispersion = false;
        std::size_t max_iter = 100;
        double tol = 0.01;
        std::size_t threads = 1;
        std::string precision = "double";
    };

    inline Options parse_args(int argc, char** argv) {
        Options opts;
        std::vector<std::string> positional;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--pangenome") {
                continue;
            }
            if (arg == "--free-dispersion") {
                opts.free_dispersion = true;
                continue;
            }
            if (arg.rfind("--", 0) == 0) {
                std::string key = arg.substr(2);
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for --" + key);
                }
                std::string value = argv[++i];
                if (key == "out-dir") {
                    opts.out_dir = value;
                } else if (key == "k") {
                    opts.k = std::stoul(value);
                } else if (key == "beta") {
                    opts.beta = std::stod(value);
                } else if (key == "sm-degree") {
                    opts.sm_degree = std::stoul(value);
                } else if (key == "max-iter") {
                    opts.max_iter = std::stoul(value);
                } else if (key == "tol") {
                    opts.tol = std::stod(value);
                } else if (key == "threads") {
                    opts.threads = std::stoul(value);
                } else if (key == "precision") {
                    opts.precision = value;
                } else {
                    throw std::runtime_error("unknown flag: --" + key);
                }
            } else {
                positional.push_back(arg);
            }
        }
        if (positional.size() != 2) {
            throw std::runtime_error(
                "usage: fastnem_cli <presence.csv> <graph.txt> [--flags...]");
        }
        opts.feature_path = positional[0];
        opts.graph_path = positional[1];
        return opts;
    }

    template<std::floating_point T>
    int run(const Options& opts) {
        Matrix<T> X_T = read_feature_matrix<T>(opts.feature_path);
        std::size_t n = X_T.cols();
        CSRGraph<T> graph = read_edge_list<T>(opts.graph_path, n);

        auto result = partition_pangenome(X_T, graph, opts.k, T(opts.beta),
                                           opts.free_dispersion, opts.sm_degree, opts.max_iter,
                                           T(opts.tol), opts.threads);
        {
            std::ofstream out(opts.out_dir + "/partition.csv");
            if (!out.is_open()) {
                throw std::runtime_error("cannot open output file: " + opts.out_dir +
                                          "/partition.csv");
            }
            for (char c : result.partition) {
                out << c << "\n";
            }
        }
        write_matrix_csv(opts.out_dir + "/membership.csv", result.membership);
        write_matrix_csv(opts.out_dir + "/centers.csv", result.centers);
        write_matrix_csv(opts.out_dir + "/dispersions.csv", result.dispersions);
        write_vector_csv(opts.out_dir + "/proportions.csv", result.proportions);
        write_criteria_csv(opts.out_dir + "/criteria.csv", result.criteria);
        std::cout << "n_iter=" << result.n_iter << " beta=" << result.beta << "\n";
        return 0;
    }

    inline int run_dispatch(const Options& opts) {
        if (opts.precision == "float") {
            return run<float>(opts);
        }
        if (opts.precision == "double") {
            return run<double>(opts);
        }
        throw std::runtime_error("unknown precision: " + opts.precision +
                                  " (expected 'float' or 'double')");
    }

}
