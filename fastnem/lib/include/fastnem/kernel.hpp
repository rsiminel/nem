#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <fastnem/core/types.hpp>
#include <fastnem/core/matrix.hpp>
#include <fastnem/core/csr_graph.hpp>
#include <fastnem/core/simd.hpp>
#include <fastnem/core/thread_pool.hpp>
#include <fastnem/core/parameters.hpp>

namespace nem {

    template<std::floating_point T>
    struct Criteria {
        T U, D, G, L, M;
    };

    template<std::floating_point T>
    class Kernel {
    public:

        static void compute_all_contexts(const CSRGraph<T>& graph, const Matrix<T>& C,
                                          Matrix<T>& out, ThreadPool* pool = nullptr) {
            dispatch(pool, 0, graph.nodes(), [&](std::size_t i) {
                compute_context_row(graph, C, i, out.row(i));
            });
        }

        static void normalize_membership(const Matrix<T>& log_pkfki, const Matrix<T>& contexts,
                                          T beta, Matrix<T>& C_out) {
            const std::size_t N = log_pkfki.rows();
            const std::size_t K = log_pkfki.cols();
            for (std::size_t i = 0; i < N; ++i) {
                auto lp_row = log_pkfki.row(i);
                auto ctx_row = contexts.row(i);
                auto out_row = C_out.row(i);

                T maxlog = lp_row[0] + beta * ctx_row[0];
                for (std::size_t k = 1; k < K; ++k) {
                    T v = lp_row[k] + beta * ctx_row[k];
                    if (v > maxlog) {
                        maxlog = v;
                    }
                }
                if (!std::isfinite(maxlog)) {
                    for (std::size_t k = 0; k < K; ++k) {
                        out_row[k] = T(1) / T(K);
                    }
                    continue;
                }
                T total = T(0);
                for (std::size_t k = 0; k < K; ++k) {
                    T e = std::exp(lp_row[k] + beta * ctx_row[k] - maxlog);
                    out_row[k] = e;
                    total += e;
                }
                total = std::max(total, DG<T>);
                for (std::size_t k = 0; k < K; ++k) {
                    out_row[k] /= total;
                }
            }
        }

        static void e_step_seq(const CSRGraph<T>& graph, const Matrix<T>& log_pkfki, T beta,
                                Matrix<T>& CM) {
            const std::size_t N = graph.nodes();
            const std::size_t K = CM.cols();
            std::vector<T> ctx(K), lognum(K);

            for (std::size_t i = 0; i < N; ++i) {
                std::fill(ctx.begin(), ctx.end(), T(0));
                auto neighbor_idx = graph.n_i(i);
                auto neighbor_w = graph.n_w(i);
                for (std::size_t p = 0; p < neighbor_idx.size(); ++p) {
                    auto c_row = CM.row(neighbor_idx[p]);
                    T w = neighbor_w[p];
                    for (std::size_t k = 0; k < K; ++k) {
                        ctx[k] += w * c_row[k];
                    }
                }

                auto lp_row = log_pkfki.row(i);
                T maxlog = -std::numeric_limits<T>::infinity();
                for (std::size_t k = 0; k < K; ++k) {
                    T val = lp_row[k] + beta * ctx[k];
                    lognum[k] = val;
                    if (val > maxlog) {
                        maxlog = val;
                    }
                }

                auto out_row = CM.row(i);
                if (!std::isfinite(maxlog)) {
                    for (std::size_t k = 0; k < K; ++k) {
                        out_row[k] = T(1) / T(K);
                    }
                    continue;
                }

                T total = T(0);
                for (std::size_t k = 0; k < K; ++k) {
                    T e = std::exp(lognum[k] - maxlog);
                    lognum[k] = e;
                    total += e;
                }

                if (total <= T(0)) {
                    for (std::size_t k = 0; k < K; ++k) {
                        out_row[k] = T(1) / T(K);
                    }
                    continue;
                }

                for (std::size_t k = 0; k < K; ++k) {
                    out_row[k] = lognum[k] / total;
                }
            }
        }

        static Criteria<T> compute_criteria(const Matrix<T>& C, const Matrix<T>& log_pkfki,
                                             const Matrix<T>& contexts, T beta) {
            const std::size_t N = C.rows();
            const std::size_t K = C.cols();

            T d_crit = T(0);
            T g = T(0);
            for (std::size_t i = 0; i < N; ++i) {
                auto c_row = C.row(i);
                auto lp_row = log_pkfki.row(i);
                auto ctx_row = contexts.row(i);
                for (std::size_t k = 0; k < K; ++k) {
                    T c = c_row[k];
                    if (c > T(0)) {
                        T log_c = std::log(std::max(c, PF<T>));
                        d_crit += c * (lp_row[k] - log_c);
                    }
                    g += c * ctx_row[k];
                }
            }
            T u = d_crit + T(0.5) * beta * g;

            T l = T(0);
            for (std::size_t i = 0; i < N; ++i) {
                auto lp_row = log_pkfki.row(i);
                T maxlog = lp_row[0];
                for (std::size_t k = 1; k < K; ++k) {
                    maxlog = std::max(maxlog, lp_row[k]);
                }
                if (std::isfinite(maxlog)) {
                    T sumexp = T(0);
                    for (std::size_t k = 0; k < K; ++k) {
                        sumexp += std::exp(lp_row[k] - maxlog);
                    }
                    l += maxlog + std::log(std::max(sumexp, PF<T>));
                }
            }

            T z_sum = T(0);
            for (std::size_t i = 0; i < N; ++i) {
                auto ctx_row = contexts.row(i);
                T maxbc = beta * ctx_row[0];
                for (std::size_t k = 1; k < K; ++k) {
                    maxbc = std::max(maxbc, beta * ctx_row[k]);
                }
                T sumexp = T(0);
                for (std::size_t k = 0; k < K; ++k) {
                    sumexp += std::exp(beta * ctx_row[k] - maxbc);
                }
                z_sum += maxbc + std::log(sumexp);
            }
            T z = -z_sum;
            T m = d_crit + beta * g + z;

            return Criteria<T>{u, d_crit, g, l, m};
        }

    private:

        static void compute_context_row(const CSRGraph<T>& graph, const Matrix<T>& C,
                                         std::size_t i, std::span<T> out_row) {
            using sops = simd_ops<T>;
            const std::size_t k = out_row.size();

            std::size_t kk = 0;
            auto zero = sops::set1(T(0));
            for (; kk + sops::width <= k; kk += sops::width) {
                sops::store(&out_row[kk], zero);
            }
            for (; kk < k; ++kk) {
                out_row[kk] = T(0);
            }

            auto neighbor_idx = graph.n_i(i);
            auto neighbor_w = graph.n_w(i);
            for (std::size_t p = 0; p < neighbor_idx.size(); ++p) {
                auto c_row = C.row(neighbor_idx[p]);
                T w = neighbor_w[p];
                auto w_vec = sops::set1(w);
                kk = 0;
                for (; kk + sops::width <= k; kk += sops::width) {
                    auto acc = sops::load(&out_row[kk]);
                    auto cvals = sops::load(&c_row[kk]);
                    sops::store(&out_row[kk], sops::add(acc, sops::mul(w_vec, cvals)));
                }
                for (; kk < k; ++kk) {
                    out_row[kk] += w * c_row[kk];
                }
            }
        }
    };

}
