#include <sys/resource.h>

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fastnem/io.hpp>
#include <fastnem/ppanggolin.hpp>

namespace {

    template<std::floating_point T>
    int run_benchmark(const std::string& data_dir, std::size_t k, double beta,
                       std::size_t sm_degree, std::size_t max_iter, double tol,
                       std::size_t threads, const std::string& out_partition) {
        nem::Matrix<T> presence_T = nem::read_feature_matrix<T>(data_dir + "/presence.csv");
        nem::CSRGraph<T> graph = nem::read_edge_list<T>(data_dir + "/graph.txt", presence_T.cols());

        std::cout << "N=" << presence_T.cols() << " D=" << presence_T.rows()
                  << " edges=" << graph.edges() << "\n";

        auto t0 = std::chrono::steady_clock::now();
        auto result = nem::partition_pangenome(presence_T, graph, k, T(beta), false, sm_degree,
                                                max_iter, T(tol), threads);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();

        struct rusage usage {};
        getrusage(RUSAGE_SELF, &usage);
        double peak_mib = static_cast<double>(usage.ru_maxrss) / 1024.0;  // ru_maxrss is KB on Linux

        std::cout << "n_iter=" << result.n_iter << "  time=" << dt << "s  peak_rss=" << peak_mib
                  << " MiB\n";

        if (!out_partition.empty()) {
            std::ofstream out(out_partition);
            if (!out.is_open()) {
                throw std::runtime_error("cannot open output file: " + out_partition);
            }
            for (char c : result.partition) {
                out << c << "\n";
            }
        }
        return 0;
    }

}

int main(int argc, char** argv) {
    std::string data_dir = "benchs/data";
    std::size_t k = 3;
    double beta = 2.5;
    std::size_t sm_degree = 10;
    std::size_t max_iter = 100;
    double tol = 1e-4;
    std::string precision = "double";
    std::size_t threads = 1;
    std::string out_partition;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("missing value for " + arg);
                }
                return argv[++i];
            };
            if (arg == "--data-dir") {
                data_dir = next();
            } else if (arg == "--k") {
                k = std::stoul(next());
            } else if (arg == "--beta") {
                beta = std::stod(next());
            } else if (arg == "--sm-degree") {
                sm_degree = std::stoul(next());
            } else if (arg == "--max-iter") {
                max_iter = std::stoul(next());
            } else if (arg == "--tol") {
                tol = std::stod(next());
            } else if (arg == "--precision") {
                precision = next();
            } else if (arg == "--threads") {
                threads = std::stoul(next());
            } else if (arg == "--out-partition") {
                out_partition = next();
            } else {
                throw std::runtime_error("unknown flag: " + arg);
            }
        }

        if (precision == "float") {
            return run_benchmark<float>(data_dir, k, beta, sm_degree, max_iter, tol, threads,
                                         out_partition);
        }
        if (precision == "double") {
            return run_benchmark<double>(data_dir, k, beta, sm_degree, max_iter, tol, threads,
                                          out_partition);
        }
        throw std::runtime_error("unknown precision: " + precision);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
