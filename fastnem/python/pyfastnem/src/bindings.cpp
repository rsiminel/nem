#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <fastnem/ppanggolin.hpp>

namespace nb = nanobind;

namespace {

    template<class T>
    using vec_in = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

    template<std::floating_point T>
    using mat_in = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

    template<std::floating_point T>
    nb::dict partition_pangenome_impl(
        mat_in<T> presence, vec_in<int64_t> edges_from, vec_in<int64_t> edges_to,
        vec_in<T> edges_weight, std::size_t n_nodes, std::size_t k, double beta,
        bool free_dispersion, std::size_t sm_degree, std::size_t max_iter, double tol,
        std::size_t n_threads, std::optional<vec_in<T>> genome_weights,
        std::optional<vec_in<T>> completeness) {
        const std::size_t n_genes = presence.shape(0);
        const std::size_t n_org = presence.shape(1);

        nem::Matrix<T> X_T(n_org, n_genes);
        for (std::size_t i = 0; i < n_genes; ++i) {
            const T* row = presence.data() + i * n_org;
            for (std::size_t d = 0; d < n_org; ++d) {
                X_T(d, i) = row[d];
            }
        }

        const std::size_t n_edges = edges_from.shape(0);
        std::vector<nem::Edge<T>> edges(n_edges);
        for (std::size_t e = 0; e < n_edges; ++e) {
            edges[e] = nem::Edge<T>{static_cast<std::size_t>(edges_from(e)),
                                     static_cast<std::size_t>(edges_to(e)), edges_weight(e)};
        }
        nem::CSRGraph<T> graph(n_nodes, edges);

        std::vector<T> gw_storage;
        const std::vector<T>* gw_ptr = nullptr;
        if (genome_weights) {
            gw_storage.assign(genome_weights->data(), genome_weights->data() + genome_weights->shape(0));
            gw_ptr = &gw_storage;
        }

        std::vector<T> comp_storage;
        const std::vector<T>* comp_ptr = nullptr;
        if (completeness) {
            comp_storage.assign(completeness->data(), completeness->data() + completeness->shape(0));
            comp_ptr = &comp_storage;
        }

        nem::PangenomeResult<T> result = [&] {
            nb::gil_scoped_release release;
            return nem::partition_pangenome<T>(X_T, graph, k, T(beta), free_dispersion, sm_degree,
                                                max_iter, T(tol), n_threads, gw_ptr, comp_ptr);
        }();

        const std::size_t out_n = result.partition.size();
        const std::size_t out_k = result.proportions.size();
        const std::size_t out_org = result.centers.cols();

        std::vector<T> membership_flat(result.membership.data(),
                                        result.membership.data() + out_n * out_k);
        std::vector<T> centers_flat(result.centers.data(), result.centers.data() + out_k * out_org);
        std::vector<T> dispersions_flat(result.dispersions.data(),
                                         result.dispersions.data() + out_k * out_org);

        nb::dict out;
        out["partition"] = std::string(result.partition.begin(), result.partition.end());
        out["membership_flat"] = membership_flat;
        out["labels_index"] = result.labels;
        out["centers_flat"] = centers_flat;
        out["dispersions_flat"] = dispersions_flat;
        out["proportions"] = result.proportions;
        out["beta"] = double(result.beta);
        out["n_iter"] = result.n_iter;
        out["k"] = out_k;
        out["n_org"] = out_org;
        out["criteria"] = std::vector<double>{double(result.criteria.U), double(result.criteria.D),
                                               double(result.criteria.G), double(result.criteria.L),
                                               double(result.criteria.M)};
        if (result.completeness) {
            out["completeness"] = std::vector<T>(*result.completeness);
        }
        return out;
    }

}

#ifndef FASTNEM_MODULE_NAME
#  define FASTNEM_MODULE_NAME _core
#endif

#define NB_MODULE_WRAP(name, m) NB_MODULE(name, m)

NB_MODULE_WRAP(FASTNEM_MODULE_NAME, m) {
    m.doc() = "bindings for fastnem";

    m.def("partition_pangenome_f64", &partition_pangenome_impl<double>, nb::arg("presence"),
          nb::arg("edges_from"), nb::arg("edges_to"), nb::arg("edges_weight"), nb::arg("n_nodes"),
          nb::arg("k"), nb::arg("beta"), nb::arg("free_dispersion"), nb::arg("sm_degree"),
          nb::arg("max_iter"), nb::arg("tol"), nb::arg("n_threads"),
          nb::arg("genome_weights") = nb::none(), nb::arg("completeness") = nb::none());

    m.def("partition_pangenome_f32", &partition_pangenome_impl<float>, nb::arg("presence"),
          nb::arg("edges_from"), nb::arg("edges_to"), nb::arg("edges_weight"), nb::arg("n_nodes"),
          nb::arg("k"), nb::arg("beta"), nb::arg("free_dispersion"), nb::arg("sm_degree"),
          nb::arg("max_iter"), nb::arg("tol"), nb::arg("n_threads"),
          nb::arg("genome_weights") = nb::none(), nb::arg("completeness") = nb::none());
}
