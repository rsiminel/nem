#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <vector>

#include <fastnem/core/csr_graph.hpp>

TEMPLATE_TEST_CASE("CSRGraph",
                    "[csr_graph]", float, double) {
    using nem::Edge;
    std::vector<Edge<TestType>> edges = {
        {0, 1, TestType(2.0)},
        {0, 2, TestType(3.0)},
        {1, 0, TestType(2.0)},
        {2, 0, TestType(3.0)},
    };
    nem::CSRGraph<TestType> g(3, edges);

    REQUIRE(g.nodes() == 3);
    REQUIRE(g.edges() == 4);

    auto idx0 = g.n_i(0);
    auto w0 = g.n_w(0);
    REQUIRE(idx0.size() == 2);
    REQUIRE(w0.size() == 2);
    REQUIRE(idx0[0] == 1);
    REQUIRE(idx0[1] == 2);
    REQUIRE(w0[0] == TestType(2.0));
    REQUIRE(w0[1] == TestType(3.0));

    auto idx1 = g.n_i(1);
    REQUIRE(idx1.size() == 1);
    REQUIRE(idx1[0] == 0);
}
