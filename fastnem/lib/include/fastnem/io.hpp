#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fastnem/core/matrix.hpp>
#include <fastnem/core/csr_graph.hpp>
#include <fastnem/kernel.hpp>

namespace nem {

    namespace detail_io {

        inline std::string trim(const std::string& s) {
            auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) {
                return "";
            }
            auto end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        }

        inline bool iequals(const std::string& a, const std::string& b) {
            if (a.size() != b.size()) {
                return false;
            }
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (std::tolower(static_cast<unsigned char>(a[i])) !=
                    std::tolower(static_cast<unsigned char>(b[i]))) {
                    return false;
                }
            }
            return true;
        }

        template<typename T>
        T parse_cell(const std::string& raw) {
            std::string tok = trim(raw);
            if (tok.empty() || iequals(tok, "nan")) {
                return std::numeric_limits<T>::quiet_NaN();
            }
            return static_cast<T>(std::stod(tok));
        }

    }

    // Reads a dense CSV feature matrix (N rows x D cols, comma-separated, no
    // header; an empty cell or "nan"/"NaN" means missing) and returns it
    // transposed, shape (D, N) matching the feature-major layout every
    // engine kernel expects.
    template<std::floating_point T>
    Matrix<T> read_feature_matrix(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("cannot open feature file: " + path);
        }

        std::vector<std::vector<T>> rows;
        std::string line;
        std::size_t d = 0;
        while (std::getline(in, line)) {
            if (detail_io::trim(line).empty()) {
                continue;
            }
            std::vector<T> row;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                row.push_back(detail_io::parse_cell<T>(cell));
            }
            if (rows.empty()) {
                d = row.size();
            } else if (row.size() != d) {
                throw std::runtime_error("inconsistent column count in feature file: " + path);
            }
            rows.push_back(std::move(row));
        }
        if (rows.empty()) {
            throw std::runtime_error("feature file has no data rows: " + path);
        }

        std::size_t n = rows.size();
        Matrix<T> X_T(d, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < d; ++j) {
                X_T(j, i) = rows[i][j];
            }
        }
        return X_T;
    }

    // Reads a directed weighted edge list: one edge per line, whitespace-
    // separated "i j [weight]" (weight optional, default 1.0)
    // blank lines and lines starting with '#' are skipped.
    template<std::floating_point T>
    CSRGraph<T> read_edge_list(const std::string& path, std::size_t n_nodes) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("cannot open graph file: " + path);
        }

        std::vector<Edge<T>> edges;
        std::string line;
        while (std::getline(in, line)) {
            std::string t = detail_io::trim(line);
            if (t.empty() || t[0] == '#') {
                continue;
            }
            std::stringstream ss(t);
            std::size_t from, to;
            if (!(ss >> from >> to)) {
                throw std::runtime_error("malformed edge line in " + path + ": " + line);
            }
            T weight = T(1);
            T maybe_weight;
            if (ss >> maybe_weight) {
                weight = maybe_weight;
            }
            edges.push_back(Edge<T>{from, to, weight});
        }
        return CSRGraph<T>(n_nodes, edges);
    }

    template<std::floating_point T>
    void write_matrix_csv(const std::string& path, const Matrix<T>& M) {
        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("cannot open output file: " + path);
        }
        for (std::size_t i = 0; i < M.rows(); ++i) {
            auto row = M.row(i);
            for (std::size_t j = 0; j < row.size(); ++j) {
                if (j) {
                    out << ",";
                }
                out << row[j];
            }
            out << "\n";
        }
    }

    template<typename T>
    void write_vector_csv(const std::string& path, const std::vector<T>& v) {
        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("cannot open output file: " + path);
        }
        for (const auto& x : v) {
            out << x << "\n";
        }
    }

    template<std::floating_point T>
    void write_criteria_csv(const std::string& path, const Criteria<T>& c) {
        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("cannot open output file: " + path);
        }
        out << "U," << c.U << "\n";
        out << "D," << c.D << "\n";
        out << "G," << c.G << "\n";
        out << "L," << c.L << "\n";
        out << "M," << c.M << "\n";
    }

}
