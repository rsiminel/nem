#pragma once

#include <cstddef>
#include <simde/x86/avx2.h>

namespace nem {

    template<std::floating_point T>
    struct simd_ops;

    template<>
    struct simd_ops<float> {
        using vec_t = simde__m256;
        static constexpr std::size_t width = 8;

        static vec_t load(const float* p) {
            return simde_mm256_loadu_ps(p);
        }

        static void store(float* p, vec_t v) {
            return simde_mm256_storeu_ps(p, v);
        }

        static vec_t set1(float x) {
            return simde_mm256_set1_ps(x);
        }

        static vec_t add(vec_t a, vec_t b) {
            return simde_mm256_add_ps(a, b);
        }

        static vec_t sub(vec_t a, vec_t b) {
            return simde_mm256_sub_ps(a, b);
        }

        static vec_t mul(vec_t a, vec_t b) {
            return simde_mm256_mul_ps(a, b);
        }

        static vec_t div(vec_t a, vec_t b) {
            return simde_mm256_div_ps(a, b);
        }

        static vec_t blend(vec_t a, vec_t b, vec_t mask) {
            return simde_mm256_blendv_ps(a, b, mask);
        }

        static vec_t abs(vec_t x) {
            return simde_mm256_andnot_ps(simde_mm256_set1_ps(-0.0f), x);
        }

        static vec_t isnan_mask(vec_t x) {
            return simde_mm256_cmp_ps(x, x, SIMDE_CMP_UNORD_Q);
        }
   };

    template<>
    struct simd_ops<double> {
        using vec_t = simde__m256d;
        static constexpr std::size_t width = 4;

        static vec_t load(const double* p) {
            return simde_mm256_loadu_pd(p);
        }

        static void store(double* p, vec_t v) {
            return simde_mm256_storeu_pd(p, v);
        }

        static vec_t set1(double x) {
            return simde_mm256_set1_pd(x);
        }

        static vec_t add(vec_t a, vec_t b) {
            return simde_mm256_add_pd(a, b);
        }

        static vec_t sub(vec_t a, vec_t b) {
            return simde_mm256_sub_pd(a, b);
        }

        static vec_t mul(vec_t a, vec_t b) {
            return simde_mm256_mul_pd(a, b);
        }

        static vec_t div(vec_t a, vec_t b) {
            return simde_mm256_div_pd(a, b);
        }

        static vec_t blend(vec_t a, vec_t b, vec_t mask) {
            return simde_mm256_blendv_pd(a, b, mask);
        }

        static vec_t abs(vec_t x) {
            return simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0f), x);
        }

        static vec_t isnan_mask(vec_t x) {
            return simde_mm256_cmp_pd(x, x, SIMDE_CMP_UNORD_Q);
        }
   };

}
