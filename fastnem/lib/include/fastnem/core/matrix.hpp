#pragma once

#include <cstddef>
#include <span>

#include <fastnem/core/aligned_vector.hpp>

namespace nem {

    template<std::floating_point T>
    class Matrix {
    public:

        Matrix(std::size_t rows, std::size_t cols)
            : m_rows(rows), m_cols(cols), m_data(rows * cols) {}

        Matrix(std::size_t rows, std::size_t cols, const T& fill)
            : m_rows(rows), m_cols(cols), m_data(rows * cols, fill) {}

        std::size_t rows() const noexcept {
            return m_rows;
        }

        std::size_t cols() const noexcept {
            return m_cols;
        }

        T& operator()(std::size_t r, std::size_t c) noexcept {
            return m_data[r * m_cols + c];
        }

        const T& operator()(std::size_t r, std::size_t c) const noexcept {
            return m_data[r * m_cols + c];
        }

        std::span<T> row(std::size_t r) noexcept {
            return std::span<T>(m_data.data() + r * m_cols, m_cols);
        }

        std::span<const T> row(std::size_t r) const noexcept {
            return std::span<const T>(m_data.data() + r * m_cols, m_cols);
        }

        T* data() noexcept {
            return m_data.data();
        }

        const T* data() const noexcept {
            return m_data.data();
        }

        Matrix clone() const {
            Matrix out(m_rows, m_cols);

            for (std::size_t i = 0; i < m_rows * m_cols; ++i) {
                out.m_data[i] = m_data[i];
            }
        }


    private:
        std::size_t m_rows {0};
        std::size_t m_cols {0};
        AlignedVector<T> m_data;

    };
}
