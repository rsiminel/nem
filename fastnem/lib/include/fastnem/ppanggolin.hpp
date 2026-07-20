#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <fastnem/core/types.hpp>
#include <fastnem/core/matrix.hpp>
#include <fastnem/core/csr_graph.hpp>
#include <fastnem/core/parameters.hpp>
#include <fastnem/engine.hpp>

namespace nem {

    template<std::floating_point T>
    Parameters<T> build_param_init(std::size_t K, std::size_t n_org) {
        T p = T(std::round((T(1) / T(K)) * T(100)) / T(100));
        std::vector<T> proportions(K);
        for (std::size_t k = 0; k + 1 < K; ++k) {
            proportions[k] = p;
        }
        proportions[K - 1] = T(1) - p * T(K - 1);

        Matrix<T> centers(K, n_org);
        Matrix<T> dispersions(K, n_org);
        T step = T(0.5) / T(std::ceil(T(K) / T(2)));
        T pichenette = (K == 2) ? T(0.1) : T(0);

        for (std::size_t k1 = 1; k1 <= K; ++k1) {
            T mu, eps;
            if (T(k1) <= T(K) / T(2)) {
                mu = T(1);
                eps = step * T(k1) - pichenette;
            } else {
                mu = T(0);
                eps = step * T(K - k1 + 1) - pichenette;
            }
            for (std::size_t d = 0; d < n_org; ++d) {
                centers(k1 - 1, d) = mu;
                dispersions(k1 - 1, d) = eps;
            }
        }

        return Parameters<T>{std::move(centers), std::move(dispersions), std::move(proportions)};
    }

    template<std::floating_point T>
    std::pair<CSRGraph<T>, T> build_nem_neighborhood(const CSRGraph<T>& graph,
                                                       std::size_t sm_degree = 10) {
        const std::size_t n = graph.nodes();
        std::vector<Edge<T>> edges;
        T total = T(0);
        for (std::size_t i = 0; i < n; ++i) {
            auto nbr_idx = graph.n_i(i);
            auto nbr_w = graph.n_w(i);
            std::size_t deg = nbr_idx.size();
            if (deg > 0 && deg < sm_degree) {
                for (std::size_t p = 0; p < deg; ++p) {
                    edges.push_back(Edge<T>{i, nbr_idx[p], nbr_w[p]});
                    total += nbr_w[p];
                }
            }
        }
        CSRGraph<T> H(n, edges);
        return {std::move(H), total / T(2)};
    }

    template<std::floating_point T>
    char assign_partition(std::span<const T> c_i, std::size_t K) {
        T mx = c_i[0];
        for (std::size_t k = 1; k < K; ++k) {
            mx = std::max(mx, c_i[k]);
        }
        std::size_t count_max = 0;
        std::size_t first_max = 0;
        for (std::size_t k = 0; k < K; ++k) {
            if (c_i[k] == mx) {
                if (count_max == 0) {
                    first_max = k;
                }
                ++count_max;
            }
        }
        if (count_max > 1 || mx < T(0.5)) {
            return 'S';
        }
        if (first_max == 0) {
            return 'P';
        }
        if (first_max == K - 1) {
            return 'C';
        }
        return 'S';
    }

    template<std::floating_point T>
    struct PangenomeResult {
        std::vector<char> partition;       // (N,) 'P'/'S'/'C' per gene family
        Matrix<T> membership;              // (N, K)
        std::vector<std::size_t> labels;   // (N,), 1-based, 0=persistent index
        Matrix<T> centers;                 // (K, n_org)
        Matrix<T> dispersions;             // (K, n_org)
        std::vector<T> proportions;        // (K,)
        T beta;                            // rescaled beta actually used
        Criteria<T> criteria;
        std::size_t n_iter;
        std::optional<std::vector<T>> completeness = std::nullopt;
        std::optional<std::size_t> completeness_n_iter = std::nullopt;
    };

    template<std::floating_point T>
    PangenomeResult<T> partition_pangenome(const Matrix<T>& presence_T, const CSRGraph<T>& graph,
                                            std::size_t K = 3, T beta = T(2.5),
                                            bool free_dispersion = false, std::size_t sm_degree = 10,
                                            std::size_t max_iter = 100, T tol = T(0.01),
                                            std::size_t n_threads = 1,
                                            const std::vector<T>* genome_weights = nullptr,
                                            const std::vector<T>* completeness = nullptr) {
        const std::size_t N = presence_T.cols();
        const std::size_t n_org = presence_T.rows();

        auto [H, half_total_edge_weight] = build_nem_neighborhood(graph, sm_degree);
        T beta_scaled = (half_total_edge_weight <= T(0))
                            ? beta
                            : beta * (T(N) / half_total_edge_weight);

        Parameters<T> init_params = build_param_init<T>(K, n_org);

        FitConfig<T> config;
        config.n_clusters = K;
        config.beta = beta_scaled;
        config.family = FamilyKind::Bernoulli;
        config.dispersion = free_dispersion ? Dispersion::Skd : Dispersion::Sk_;
        config.proportion = Proportion::Free;
        config.algorithm = Algorithm::Nem;
        config.site_update = SiteUpdate::Seq;
        config.missing = Missing::Ignore;
        config.convergence = Convergence::Classification;
        config.init = Init::Param;
        config.init_params = std::move(init_params);
        if (genome_weights != nullptr) {
            config.feature_weights = std::vector<T>(*genome_weights);
        }
        if (completeness != nullptr) {
            config.completeness = std::vector<T>(*completeness);
        }
        config.max_iter = max_iter;
        config.tol = tol;
        config.n_threads = n_threads;

        FitResult<T> fit_result = fit(presence_T, H, config);

        std::vector<char> partition(N);
        for (std::size_t i = 0; i < N; ++i) {
            partition[i] = assign_partition<T>(fit_result.membership.row(i), K);
        }

        PangenomeResult<T> result{std::move(partition),
                                   std::move(fit_result.membership),
                                   std::move(fit_result.labels),
                                   std::move(fit_result.params.centers),
                                   std::move(fit_result.params.dispersions),
                                   std::move(fit_result.params.proportions),
                                   beta_scaled,
                                   fit_result.criteria,
                                   fit_result.n_iter};
        if (completeness != nullptr) {
            result.completeness = std::vector<T>(*completeness);
        }
        return result;
    }

    template<std::floating_point T>
    PangenomeResult<T> partition_pangenome_self_completeness(
        const Matrix<T>& presence_T, const CSRGraph<T>& graph, std::size_t K = 3, T beta = T(2.5),
        bool free_dispersion = false, std::size_t sm_degree = 10, std::size_t max_iter = 100,
        T tol = T(0.01), std::size_t n_threads = 1, const std::vector<T>* genome_weights = nullptr,
        std::size_t comp_iters = 15, T comp_tol = T(1e-3), T gmin = T(0.05)) {
        const std::size_t N = presence_T.cols();
        const std::size_t n_org = presence_T.rows();

        std::vector<T> sizes(n_org, T(0));
        for (std::size_t d = 0; d < n_org; ++d) {
            auto row = presence_T.row(d);
            for (std::size_t i = 0; i < N; ++i) {
                sizes[d] += row[i];
            }
        }
        T max_size = T(1);
        for (std::size_t d = 0; d < n_org; ++d) {
            max_size = std::max(max_size, sizes[d]);
        }

        std::vector<T> gamma(n_org);
        for (std::size_t d = 0; d < n_org; ++d) {
            gamma[d] = std::clamp(sizes[d] / max_size, gmin, T(1));
        }

        std::optional<PangenomeResult<T>> result;
        std::size_t n_round = 0;
        for (std::size_t round = 1; round <= comp_iters; ++round) {
            n_round = round;
            result = partition_pangenome(presence_T, graph, K, beta, free_dispersion, sm_degree,
                                          max_iter, tol, n_threads, genome_weights, &gamma);

            std::size_t persistent_count = 0;
            for (char p : result->partition) {
                if (p == 'P') {
                    ++persistent_count;
                }
            }
            if (persistent_count == 0) {
                break;
            }

            std::vector<T> new_gamma(n_org, T(0));
            for (std::size_t d = 0; d < n_org; ++d) {
                auto row = presence_T.row(d);
                T sum = T(0);
                for (std::size_t i = 0; i < N; ++i) {
                    if (result->partition[i] == 'P') {
                        sum += row[i];
                    }
                }
                new_gamma[d] = std::clamp(sum / T(persistent_count), gmin, T(1));
            }

            T delta = T(0);
            for (std::size_t d = 0; d < n_org; ++d) {
                delta = std::max(delta, std::abs(new_gamma[d] - gamma[d]));
            }
            gamma = new_gamma;

            if (delta < comp_tol) {
                break;
            }
        }

        result->completeness = gamma;
        result->completeness_n_iter = n_round;
        return std::move(*result);
    }

}
