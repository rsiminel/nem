#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <fastnem/core/aligned_vector.hpp>

namespace nem {

    template<std::floating_point T>
    struct Edge {
        std::size_t from {0};
        std::size_t to {0};
        T weight {0.0};
    };

    template<std::floating_point T>
    class CSRGraph {
    public:

        CSRGraph(std::size_t n_nodes, const std::vector<Edge<T>>& edges)
            : m_n_nodes(n_nodes), m_ip(n_nodes + 1, 0), m_i(edges.size()), m_data(edges.size()) {

            for (const auto& e : edges) {
                m_ip[e.from + 1] += 1;
            }

            for (std::size_t i = 0; i < m_n_nodes; ++i) {
                m_ip[i + 1] += m_ip[i];
            }

            std::vector<std::size_t> c(m_ip.begin(), m_ip.end() - 1);

            for (const auto& e : edges) {
                std::size_t p = c[e.from]++;
                m_i[p] = e.to;
                m_data[p] = e.weight;
            }
        }

        std::size_t nodes() const noexcept {
            return m_n_nodes;
        }

        std::size_t edges() const noexcept {
            return m_i.size();
        }

        std::span<const std::size_t> n_i(std::size_t i) const noexcept {
            return std::span<const std::size_t>(
                m_i.data() + m_ip[i], m_ip[i + 1] - m_ip[i]
            );
        }

        std::span<const T> n_w(std::size_t i) const noexcept {
            return std::span<const T>(
                m_data.data() + m_ip[i], m_ip[i + 1] - m_ip[i]
            );
        }


    private:
        std::size_t m_n_nodes {0};
        std::vector<std::size_t> m_ip;
        std::vector<std::size_t> m_i;
        AlignedVector<T> m_data;
    };
}
