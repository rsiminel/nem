#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <fastnem/core/types.hpp>
#include <fastnem/core/matrix.hpp>
#include <fastnem/core/csr_graph.hpp>
#include <fastnem/core/thread_pool.hpp>
#include <fastnem/core/parameters.hpp>
#include <fastnem/family.hpp>
#include <fastnem/kernel.hpp>
#include <fastnem/convergence.hpp>

namespace nem {

    template<std::floating_point T>
    struct FitConfig {
        std::size_t n_clusters;
        T beta;
        BetaMode beta_mode = BetaMode::Fix;
        FamilyKind family = FamilyKind::Bernoulli;
        Algorithm algorithm = Algorithm::Nem;
        Dispersion dispersion = Dispersion::Sk_;
        Proportion proportion = Proportion::Free;
        SiteUpdate site_update = SiteUpdate::Seq;
        Missing missing = Missing::Ignore;
        Convergence convergence = Convergence::Classification;
        Init init = Init::Param;
        std::optional<Parameters<T>> init_params = std::nullopt;
        std::optional<std::vector<T>> feature_weights = std::nullopt;
        std::optional<std::vector<T>> completeness = std::nullopt;
        std::size_t max_iter = 100;
        T tol = T(1e-3);
        std::size_t n_threads = 1;
    };

    template<std::floating_point T>
    struct FitResult {
        Matrix<T> membership;             // (N, K)
        std::vector<std::size_t> labels;  // (N,), 1-based
        Parameters<T> params;
        T beta;
        Criteria<T> criteria;
        std::size_t n_iter;
        std::vector<Criteria<T>> history;
    };

    namespace detail {

        template<std::floating_point T>
        std::unique_ptr<Family<T>> make_config_family(const FitConfig<T>& config) {
            const std::vector<T>* w = config.feature_weights ? &(*config.feature_weights) : nullptr;
            const std::vector<T>* c = config.completeness ? &(*config.completeness) : nullptr;
            return make_family<T>(config.family, config.dispersion, config.proportion,
                                   config.missing, w, c);
        }

        template<std::floating_point T>
        FitResult<T> fit_from_classification(const Matrix<T>& X_T, const CSRGraph<T>& graph,
                                              const FitConfig<T>& config, Matrix<T> C,
                                              ThreadPool& pool) {
            const std::size_t N = X_T.cols();
            const std::size_t K = config.n_clusters;

            std::unique_ptr<Family<T>> family = make_config_family(config);

            Matrix<T> log_pkfki(N, K);
            Matrix<T> contexts(N, K);

            std::optional<Parameters<T>> params;
            std::optional<Criteria<T>> old_crit;
            std::vector<Criteria<T>> history;
            std::size_t n_iter = 0;
            T beta = config.beta;

            for (std::size_t iter = 0; iter < config.max_iter; ++iter) {
                params = family->estimate(X_T, C, params ? &(*params) : nullptr, &pool);

                family->density(X_T, params->centers, params->dispersions, params->proportions,
                                 log_pkfki);

                Matrix<T> old_C = C.clone();
                Kernel<T>::e_step_seq(graph, log_pkfki, beta, C);

                Kernel<T>::compute_all_contexts(graph, C, contexts, &pool);
                Criteria<T> criteria = Kernel<T>::compute_criteria(C, log_pkfki, contexts, beta);
                history.push_back(criteria);
                n_iter = iter + 1;

                bool converged = old_crit.has_value() && has_converged(old_C, C, config.tol);
                old_crit = criteria;
                if (converged) {
                    break;
                }
            }

            std::vector<std::size_t> labels(N);
            for (std::size_t i = 0; i < N; ++i) {
                auto row = C.row(i);
                std::size_t best = 0;
                T bestval = row[0];
                for (std::size_t k = 1; k < K; ++k) {
                    if (row[k] > bestval) {
                        bestval = row[k];
                        best = k;
                    }
                }
                labels[i] = best + 1;
            }

            return FitResult<T>{std::move(C),      std::move(labels), std::move(*params),
                                 beta,              *old_crit,         n_iter,
                                 std::move(history)};
        }

    }

    template<std::floating_point T>
    FitResult<T> fit(const Matrix<T>& X_T, const CSRGraph<T>& graph, const FitConfig<T>& config) {
        const std::size_t N = X_T.cols();
        const std::size_t K = config.n_clusters;

        ThreadPool pool(config.n_threads);

        Matrix<T> C(N, K);
        std::unique_ptr<Family<T>> family = detail::make_config_family(config);

        const Parameters<T>& ip = *config.init_params;
        Matrix<T> floored_disp = ip.dispersions.clone();
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t d = 0; d < floored_disp.cols(); ++d) {
                floored_disp(k, d) = std::max(floored_disp(k, d), VF<T>);
            }
        }

        Matrix<T> log_pkfki_init(N, K);
        family->density(X_T, ip.centers, floored_disp, ip.proportions, log_pkfki_init);

        Matrix<T> zero_context(N, K, T(0));
        Kernel<T>::normalize_membership(log_pkfki_init, zero_context, T(0), C);

        Kernel<T>::e_step_seq(graph, log_pkfki_init, config.beta, C);

        return detail::fit_from_classification(X_T, graph, config, std::move(C), pool);
    }

}
