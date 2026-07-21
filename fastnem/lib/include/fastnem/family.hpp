#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <fastnem/core/types.hpp>
#include <fastnem/core/matrix.hpp>
#include <fastnem/core/simd.hpp>
#include <fastnem/core/thread_pool.hpp>
#include <fastnem/core/parameters.hpp>

namespace nem {

template<std::floating_point T>
class Family {
public:
    Family(Dispersion d, Proportion p, Missing m, const std::vector<T>* w = nullptr)
     : m_d(d), m_p(p), m_m(m), m_weights(w) {}

    virtual ~Family() = default;

    virtual void density(const Matrix<T>& X_T,
                         const Matrix<T>& centers,
                         const Matrix<T>& dispersions,
                         const std::vector<T>& proportions,
                         Matrix<T>& log_pkfki,
                         ThreadPool* pool) const = 0;

    virtual Parameters<T> estimate(const Matrix<T>& X_T,
                          const Matrix<T>& C,
                          const Parameters<T>* old_params,
                          ThreadPool* pool) const = 0;

protected:

    static T class_log_prior(T proportion) {
        return std::log(std::max(proportion, PF<T>));
    }

    static bool is_degenerate(T v) {
        return v <= ZD<T>;
    }

    static void mark_invalid(std::span<const T> x_col, T mean, std::vector<std::uint8_t>& invalid) {
        for (std::size_t i = 0; i < x_col.size(); ++i) {
            T x = x_col[i];
            if (std::isnan(x)) {
                continue;
            }
            if (std::abs(x - mean) > ZD<T>) {
                invalid[i] = 1;
            }
        }
    }

    static void class_log_density(std::size_t k, T log_pk, const std::vector<T>& log_fki,
                                  const std::vector<std::uint8_t>& invalid,
                                  Matrix<T>& log_pkfki) {
        for (std::size_t i = 0; i < log_fki.size(); ++i) {
            log_pkfki(i, k) = invalid[i] ? -std::numeric_limits<T>::infinity()
                                          : log_pk + log_fki[i];
        }
    }

    static std::pair<std::vector<T>, std::vector<T>> class_sizes(const Matrix<T>& C) {
        const std::size_t N = C.rows();
        const std::size_t K = C.cols();

        std::vector<T> raw_n_k(K, T(0));

        for (std::size_t i = 0; i < N; ++i) {
            auto c_row = C.row(i);
            for (std::size_t k = 0; k < K; ++k) {
                raw_n_k[k] += c_row[k];
            }
        }

        std::vector<T> n_k(K);

        for (std::size_t k = 0; k < K; ++k) {
            n_k[k] = std::max(raw_n_k[k], DG<T>);
        }

        return {raw_n_k, n_k};
    }

    std::vector<T> proportions(const std::vector<T>& n_k, std::size_t N) const {
        const std::size_t K = n_k.size();
        std::vector<T> proportions(K);
        T total = T(0);
        for (std::size_t k = 0; k < K; ++k) {
            proportions[k] = std::max(n_k[k] / T(N), PF<T>);
            total += proportions[k];
        }
        for (std::size_t k = 0; k < K; ++k) {
            proportions[k] /= total;
        }
        return proportions;
    }

    static void inertia_to_dispersions(const Matrix<T>& inertia_kd, const Matrix<T>& n_kd,
                                        std::size_t d_total, std::size_t k_total, Dispersion model,
                                        const std::vector<T>* weights, Matrix<T>& dispersions_out) {
        auto w_at = [&](std::size_t d) { return weight_at(weights, d); };

        switch (model) {
            case Dispersion::Sk_: {
                for (std::size_t k = 0; k < k_total; ++k) {
                    T num = T(0);
                    for (std::size_t d = 0; d < d_total; ++d) {
                        num += w_at(d) * inertia_kd(k, d);
                    }
                    T den = T(0);
                    for (std::size_t d = 0; d < d_total; ++d) {
                        den += w_at(d) * n_kd(k, d);
                    }
                    T vk = num / den;
                    for (std::size_t d = 0; d < d_total; ++d) {
                        dispersions_out(k, d) = vk;
                    }
                }
                break;
            }
            case Dispersion::Skd: {
                for (std::size_t k = 0; k < k_total; ++k) {
                    for (std::size_t d = 0; d < d_total; ++d) {
                        dispersions_out(k, d) = inertia_kd(k, d) / n_kd(k, d);
                    }
                }
                break;
            }
        }
    }

    static void reinit_empty_classes(const Matrix<T>& X_T, const std::vector<T>& raw_n_k,
                                      std::size_t n, std::size_t d_total, std::size_t k_total,
                                      Matrix<T>& centers, Matrix<T>& dispersions,
                                      std::vector<T>& proportions) {
        std::vector<std::size_t> empty;
        for (std::size_t k = 0; k < k_total; ++k) {
            if (raw_n_k[k] < EW<T>) {
                empty.push_back(k);
            }
        }
        if (empty.empty()) {
            return;
        }

        std::vector<T> default_disp(d_total, T(0));
        std::size_t n_populated = 0;
        for (std::size_t k = 0; k < k_total; ++k) {
            if (raw_n_k[k] >= EW<T>) {
                ++n_populated;
                for (std::size_t d = 0; d < d_total; ++d) {
                    default_disp[d] += dispersions(k, d);
                }
            }
        }
        if (n_populated > 0) {
            for (std::size_t d = 0; d < d_total; ++d) {
                default_disp[d] = std::max(default_disp[d] / T(n_populated), VF<T>);
            }
        } else {
            for (std::size_t d = 0; d < d_total; ++d) {
                auto col = X_T.row(d);
                T sum = T(0), sumsq = T(0);
                std::size_t cnt = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    T x = col[i];
                    if (std::isnan(x)) {
                        continue;
                    }
                    sum += x;
                    sumsq += x * x;
                    ++cnt;
                }
                T mean = cnt > 0 ? sum / T(cnt) : T(0);
                T var = cnt > 0 ? sumsq / T(cnt) - mean * mean : T(0);
                default_disp[d] = std::max(var, VF<T>);
            }
        }

        std::size_t dominant = 0;
        for (std::size_t k = 1; k < k_total; ++k) {
            if (raw_n_k[k] > raw_n_k[dominant]) {
                dominant = k;
            }
        }

        std::vector<T> d2(n, T(0));
        for (std::size_t d = 0; d < d_total; ++d) {
            auto col = X_T.row(d);
            T c = centers(dominant, d);
            for (std::size_t i = 0; i < n; ++i) {
                T x = col[i];
                if (std::isnan(x)) {
                    continue;
                }
                T diff = x - c;
                d2[i] += diff * diff;
            }
        }

        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                          [&](std::size_t a, std::size_t b) { return d2[a] < d2[b]; });
        std::reverse(order.begin(), order.end());

        std::vector<bool> used(n, false);
        for (std::size_t k : empty) {
            std::size_t far = order[0];
            for (std::size_t idx : order) {
                if (!used[idx]) {
                    far = idx;
                    break;
                }
            }
            used[far] = true;
            for (std::size_t d = 0; d < d_total; ++d) {
                auto col = X_T.row(d);
                T x = col[far];
                centers(k, d) = std::isnan(x) ? T(0) : x;
                dispersions(k, d) = default_disp[d];
            }
            proportions[k] = std::max(proportions[k], T(1) / T(n));
        }

        T total = T(0);
        for (T p : proportions) {
            total += p;
        }
        for (T& p : proportions) {
            p /= total;
        }
    }

    Dispersion m_d;
    Proportion m_p;
    Missing m_m;
    const std::vector<T>* m_weights;
};

template<std::floating_point T>
class Bernoulli : public Family<T> {
public:
    Bernoulli(Dispersion d, Proportion p, Missing m, const std::vector<T>* weights = nullptr,
              const std::vector<T>* completeness = nullptr)
      : Family<T>(d, p, m, weights), m_completeness(completeness) {}

    void density(const Matrix<T>& X_T, const Matrix<T>& centers,
                 const Matrix<T>& dispersions, const std::vector<T>& proportions,
                 Matrix<T>& log_pkfki, ThreadPool* pool) const override {
        if (m_completeness != nullptr) {
            density_mag_aware(X_T, centers, dispersions, proportions, log_pkfki, pool);
        } else {
            density_standard(X_T, centers, dispersions, proportions, log_pkfki, pool);
        }
    }

    Parameters<T> estimate(const Matrix<T>& X_T, const Matrix<T>& C,
                           const Parameters<T>* /*old_params*/, ThreadPool* pool) const override {
        if (m_completeness != nullptr) {
            return estimate_mag_aware(X_T, C, pool);
        }
        return estimate_standard(X_T, C, pool);
    }

private:
    using Family<T>::m_d;
    using Family<T>::m_p;
    using Family<T>::m_m;
    using Family<T>::m_weights;
    using Family<T>::class_log_prior;
    using Family<T>::is_degenerate;
    using Family<T>::mark_invalid;
    using Family<T>::class_log_density;
    using Family<T>::class_sizes;
    using Family<T>::proportions;
    using Family<T>::inertia_to_dispersions;
    using Family<T>::reinit_empty_classes;

    void density_standard(const Matrix<T>& X_T, const Matrix<T>& centers,
                           const Matrix<T>& dispersions, const std::vector<T>& proportions,
                           Matrix<T>& log_pkfki, ThreadPool* pool) const {
        const std::size_t D = X_T.rows();
        const std::size_t N = X_T.cols();
        const std::size_t K = centers.rows();

        dispatch(pool, 0, K, [&](std::size_t k) {
            std::vector<T> log_fki(N, T(0));
            std::vector<std::uint8_t> invalid(N, std::uint8_t(0));

            T log_pk = class_log_prior(proportions[k]);
            auto centers_k = centers.row(k);
            auto disp_k = dispersions.row(k);

            for (std::size_t d = 0; d < D; ++d) {
                T v = disp_k[d];
                T mean = centers_k[d];
                auto x_col = X_T.row(d);
                T w_d = weight_at(m_weights, d);

                if (is_degenerate(v)) {
                    bool active_d = m_weights == nullptr || w_d > T(0);
                    if (active_d) {
                        mark_invalid(x_col, mean, invalid);
                    }
                } else {
                    accumulate_bernoulli_term_simd(x_col, mean, v, w_d, log_fki);
                }
            }

            class_log_density(k, log_pk, log_fki, invalid, log_pkfki);
        });
    }

    static void accumulate_bernoulli_term_simd(std::span<const T> x_col, T mean, T v, T weight,
                                                std::vector<T>& log_fki) {
        const std::size_t N = x_col.size();
        using sops = simd_ops<T>;
        T log_one_minus_v = std::log(T(1) - v);
        T log_ratio = std::log((T(1) - v) / v);
        auto log_one_minus_v_vec = sops::set1(log_one_minus_v);
        auto log_ratio_vec = sops::set1(log_ratio);
        auto mean_vec = sops::set1(mean);
        auto zero_vec = sops::set1(T(0));
        auto w_vec = sops::set1(weight);

        std::size_t i = 0;
        for (; i + sops::width <= N; i += sops::width) {
            auto x_vec = sops::load(&x_col[i]);
            auto nan_mask = sops::isnan_mask(x_vec);
            auto diff = sops::sub(x_vec, mean_vec);
            auto absdiff = sops::abs(diff);
            auto term = sops::sub(log_one_minus_v_vec, sops::mul(absdiff, log_ratio_vec));
            auto contrib = sops::blend(term, zero_vec, nan_mask);
            contrib = sops::mul(contrib, w_vec);
            auto acc = sops::load(&log_fki[i]);
            sops::store(&log_fki[i], sops::add(acc, contrib));
        }
        for (; i < N; ++i) {
            T x = x_col[i];
            if (std::isnan(x)) {
                continue;
            }
            T absdiff = std::abs(x - mean);
            log_fki[i] += (log_one_minus_v - absdiff * log_ratio) * weight;
        }
    }

    void density_mag_aware(const Matrix<T>& X_T, const Matrix<T>& centers,
                            const Matrix<T>& dispersions, const std::vector<T>& proportions,
                            Matrix<T>& log_pkfki, ThreadPool* pool) const {
        const std::size_t D = X_T.rows();
        const std::size_t N = X_T.cols();
        const std::size_t K = centers.rows();

        dispatch(pool, 0, K, [&](std::size_t k) {
            std::vector<T> log_fki(N, T(0));

            T log_pk = class_log_prior(proportions[k]);
            auto centers_k = centers.row(k);
            auto disp_k = dispersions.row(k);

            for (std::size_t d = 0; d < D; ++d) {
                T v = disp_k[d];
                T mean = centers_k[d];
                auto x_col = X_T.row(d);
                T w_d = weight_at(m_weights, d);
                T gamma_d = (*m_completeness)[d];

                accumulate_mag_aware_term_simd(x_col, mean, v, w_d, gamma_d, log_fki);
            }

            for (std::size_t i = 0; i < N; ++i) {
                log_pkfki(i, k) = log_pk + log_fki[i];
            }
        });
    }

    static void accumulate_mag_aware_term_simd(std::span<const T> x_col, T mean, T v, T weight,
                                                T gamma_d, std::vector<T>& log_fki) {
        const std::size_t N = x_col.size();
        bool zero_v = v <= ZD<T>;
        T safe_v = zero_v ? T(0.5) : v;
        T mu = (mean == T(1)) ? (T(1) - safe_v) : safe_v;
        T mu_eff = std::clamp(mu * gamma_d, PF<T>, T(1) - PF<T>);
        T log_mu_eff = std::log(mu_eff);
        T log_one_minus_mu_eff = std::log(T(1) - mu_eff);
        T log_diff = log_mu_eff - log_one_minus_mu_eff;

        using sops = simd_ops<T>;
        auto log_one_minus_mu_eff_vec = sops::set1(log_one_minus_mu_eff);
        auto log_diff_vec = sops::set1(log_diff);
        auto zero_vec = sops::set1(T(0));
        auto w_vec = sops::set1(weight);

        std::size_t i = 0;
        for (; i + sops::width <= N; i += sops::width) {
            auto x_vec = sops::load(&x_col[i]);
            auto nan_mask = sops::isnan_mask(x_vec);
            auto term = sops::add(log_one_minus_mu_eff_vec, sops::mul(x_vec, log_diff_vec));
            auto contrib = sops::blend(term, zero_vec, nan_mask);
            contrib = sops::mul(contrib, w_vec);
            auto acc = sops::load(&log_fki[i]);
            sops::store(&log_fki[i], sops::add(acc, contrib));
        }
        for (; i < N; ++i) {
            T x = x_col[i];
            if (std::isnan(x)) {
                continue;
            }
            log_fki[i] += (log_one_minus_mu_eff + x * log_diff) * weight;
        }
    }

    static std::pair<Matrix<T>, Matrix<T>> accumulate_observed_sums(const Matrix<T>& X_T,
                                                                      const Matrix<T>& C,
                                                                      ThreadPool* pool) {
        const std::size_t D = X_T.rows();
        const std::size_t N = X_T.cols();
        const std::size_t K = C.cols();

        Matrix<T> n_kd(K, D, T(0));
        Matrix<T> weighted_sum(K, D, T(0));
        dispatch(pool, 0, D, [&](std::size_t d) {
            auto x_col = X_T.row(d);
            for (std::size_t i = 0; i < N; ++i) {
                T x = x_col[i];
                if (std::isnan(x)) {
                    continue;
                }
                auto c_row = C.row(i);
                for (std::size_t k = 0; k < K; ++k) {
                    n_kd(k, d) += c_row[k];
                    weighted_sum(k, d) += c_row[k] * x;
                }
            }
        });
        return {std::move(n_kd), std::move(weighted_sum)};
    }

    Parameters<T> estimate_standard(const Matrix<T>& X_T, const Matrix<T>& C,
                                    ThreadPool* pool) const {
        const std::size_t D = X_T.rows();
        const std::size_t N = X_T.cols();
        const std::size_t K = C.cols();

        auto [raw_n_k, n_k] = class_sizes(C);
        auto [n_kd, weighted_sum] = accumulate_observed_sums(X_T, C, pool);

        Matrix<T> centers(K, D);
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t d = 0; d < D; ++d) {
                T w_total = n_kd(k, d);
                T frac1 = w_total > T(0) ? weighted_sum(k, d) / std::max(w_total, DG<T>) : T(0);
                centers(k, d) = frac1 > T(0.5) ? T(1) : T(0);
            }
        }

        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t d = 0; d < D; ++d) {
                n_kd(k, d) = std::max(n_kd(k, d), DG<T>);
            }
        }

        Matrix<T> inertia = accumulate_abs_inertia(X_T, C, centers, pool);

        Matrix<T> dispersions(K, D);
        inertia_to_dispersions(inertia, n_kd, D, K, m_d, m_weights, dispersions);
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t d = 0; d < D; ++d) {
                dispersions(k, d) = std::max(dispersions(k, d), VF<T>);
            }
        }

        std::vector<T> props = proportions(n_k, N);

        reinit_empty_classes(X_T, raw_n_k, N, D, K, centers, dispersions, props);

        return Parameters<T>{std::move(centers), std::move(dispersions), std::move(props)};
    }

    static Matrix<T> accumulate_abs_inertia(const Matrix<T>& X_T, const Matrix<T>& C,
                                             const Matrix<T>& centers, ThreadPool* pool) {
        const std::size_t D = X_T.rows();
        const std::size_t N = X_T.cols();
        const std::size_t K = C.cols();

        Matrix<T> inertia(K, D, T(0));
        dispatch(pool, 0, D, [&](std::size_t d) {
            auto x_col = X_T.row(d);
            for (std::size_t i = 0; i < N; ++i) {
                T x = x_col[i];
                if (std::isnan(x)) {
                    continue;
                }
                auto c_row = C.row(i);
                for (std::size_t k = 0; k < K; ++k) {
                    inertia(k, d) += c_row[k] * std::abs(x - centers(k, d));
                }
            }
        });
        return inertia;
    }

    Parameters<T> estimate_mag_aware(const Matrix<T>& X_T, const Matrix<T>& C,
                                     ThreadPool* pool) const {
        const std::size_t D = X_T.rows();
        const std::size_t N = X_T.cols();
        const std::size_t K = C.cols();

        auto [raw_n_k, n_k] = class_sizes(C);
        auto [n_kd, weighted_sum] = accumulate_observed_sums(X_T, C, pool);

        Matrix<T> centers(K, D);
        Matrix<T> dispersions(K, D);
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t d = 0; d < D; ++d) {
                T w_total = n_kd(k, d);
                T frac1 = w_total > T(0) ? weighted_sum(k, d) / std::max(w_total, DG<T>) : T(0);
                T gamma_d = std::clamp((*m_completeness)[d], VF<T>, T(1));
                T mu = std::clamp(frac1 / gamma_d, T(0), T(1));
                centers(k, d) = mu > T(0.5) ? T(1) : T(0);
                T eps = centers(k, d) == T(1) ? T(1) - mu : mu;
                dispersions(k, d) = std::clamp(eps, VF<T>, T(0.5));
            }
        }

        std::vector<T> props = proportions(n_k, N);

        reinit_empty_classes(X_T, raw_n_k, N, D, K, centers, dispersions, props);

        return Parameters<T>{std::move(centers), std::move(dispersions), std::move(props)};
    }

    const std::vector<T>* m_completeness;
};

template<typename T>
inline std::unique_ptr<Family<T>> make_family(FamilyKind kind, Dispersion d, Proportion p,
                                               Missing m, const std::vector<T>* weights = nullptr,
                                               const std::vector<T>* completeness = nullptr) {
    switch (kind) {
        case FamilyKind::Bernoulli:
            return std::make_unique<Bernoulli<T>>(d, p, m, weights, completeness);
        default:
            return nullptr;
    }
}

}
