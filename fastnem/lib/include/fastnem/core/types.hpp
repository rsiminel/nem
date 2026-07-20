#pragma once

#include <cstddef>
#include <vector>

namespace nem {

    template <typename T>
    inline constexpr T PF = T(1e-20);
    template <typename T>
    inline constexpr T VF = T(1e-20);
    template <typename T>
    inline constexpr T ZD = T(1e-20);
    template <typename T>
    inline constexpr T DG = T(1e-20);
    template <typename T>
    inline constexpr T EW = T(1.0);

    enum class FamilyKind {
        Bernoulli
    };

    enum class Dispersion {
        Sk_, Skd
    };

    enum class Proportion {
        Free
    };

    enum class Missing {
        Ignore
    };

    enum class Algorithm {
        Nem
    };

    enum class SiteUpdate {
        Seq
    };

    enum class Convergence {
        Classification
    };

    enum class Init {
        Param
    };

    enum class BetaMode {
        Fix
    };

    template <typename T>
    inline T weight_at(const std::vector<T>* weights, std::size_t d) {
        return weights ? (*weights)[d] : T(1);
    }

}
