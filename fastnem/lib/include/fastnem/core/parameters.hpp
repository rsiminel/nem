#pragma once

#include <utility>
#include <vector>

#include <fastnem/core/matrix.hpp>

namespace nem {

    template<std::floating_point T>
    struct Parameters {
        Matrix<T> centers;
        Matrix<T> dispersions;
        std::vector<T> proportions;

        Parameters(Matrix<T> c, Matrix<T> d, std::vector<T> p)
            : centers(std::move(c)), dispersions(std::move(d)), proportions(std::move(p)) {}

        Parameters(const Parameters& o)
            : centers(o.centers.clone()), dispersions(o.dispersions.clone()), proportions(o.proportions) {}
        Parameters& operator=(const Parameters& o) {
            centers = o.centers.clone();
            dispersions = o.dispersions.clone();
            proportions = o.proportions;
            return *this;
        }

        Parameters(Parameters&&) = default;
        Parameters& operator=(Parameters&&) = default;
    };

}
