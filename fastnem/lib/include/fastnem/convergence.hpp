#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <fastnem/core/matrix.hpp>

namespace nem {

    template<std::floating_point T>
    bool has_converged(const Matrix<T>& c_old, const Matrix<T>& c_new, T tol) {
        const std::size_t N = c_old.rows();
        const std::size_t K = c_old.cols();
        T maxdiff = T(0);
        for (std::size_t i = 0; i < N; ++i) {
            auto old_row = c_old.row(i);
            auto new_row = c_new.row(i);
            for (std::size_t k = 0; k < K; ++k) {
                maxdiff = std::max(maxdiff, std::abs(new_row[k] - old_row[k]));
            }
        }
        return maxdiff < tol;
    }

}
