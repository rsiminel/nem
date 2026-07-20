#pragma once

#include <cstddef>
#include <concepts>
#include <new>

namespace nem {

    template<std::floating_point T, std::size_t Alignment = 32>
    class AlignedVector {
    public:

        AlignedVector() noexcept :
            m_data(nullptr), m_size(0) {}

        explicit AlignedVector(std::size_t n)
            : m_data(allocate(n)), m_size(n) {
            std::fill(begin(), end(), T{});
        }

        AlignedVector(std::size_t n, const T& fill)
            : m_data(allocate(n)), m_size(n) {
            std::fill(begin(), end(), fill);
        }

        ~AlignedVector() {
            destroy();
        }

        AlignedVector(const AlignedVector&) = delete;
        AlignedVector& operator=(const AlignedVector&) = delete;

        AlignedVector(AlignedVector&& other) noexcept
            : m_data(other.m_data), m_size(other.m_size) {
            other.m_data = nullptr;
            other.m_size = 0;
        }

        AlignedVector& operator=(AlignedVector&& other) noexcept {
            if (this != &other) {
                destroy();
                m_data = other.m_data;
                m_size = other.m_size;
                other.m_data = nullptr;
                other.m_size = 0;
            }
            return *this;
        }

        T* begin() noexcept {
            return m_data;
        }

        T* end() noexcept {
            return m_data + m_size;
        }

        const T* begin() const noexcept {
            return m_data;
        }

        const T* end() const noexcept {
            return m_data + m_size;
        }

        T* data() noexcept {
            return m_data;
        }

        const T* data() const noexcept {
            return m_data;
        }

        std::size_t size() const noexcept {
            return m_size;
        }

        T& operator[](std::size_t i) noexcept {
            return m_data[i];
        }

        const T& operator[](std::size_t i) const noexcept {
            return m_data[i];
        }

    private:
        static T* allocate(std::size_t n) {
            if (n == 0) {
                return nullptr;
            }

            return static_cast<T*>(
                ::operator new(n * sizeof(T), std::align_val_t{Alignment})
            );
        }

        void destroy() noexcept {
            if (m_data != nullptr) {
                ::operator delete(m_data, std::align_val_t{Alignment});
            }
        }
    private:
        T* m_data {nullptr};
        std::size_t m_size {0};
    };
}
