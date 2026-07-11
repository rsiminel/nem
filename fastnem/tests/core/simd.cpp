#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <vector>

#include <fastnem/core/simd.hpp>

TEMPLATE_TEST_CASE("simd_ops", "[simd]", float, double) {
    using ops = nem::simd_ops<TestType>;
    std::vector<TestType> a(ops::width), b(ops::width), out(ops::width);
    for (std::size_t i = 0; i < ops::width; ++i) {
        a[i] = TestType(i);
        b[i] = TestType(10);
    }

    auto va = ops::load(a.data());
    auto vb = ops::load(b.data());
    auto vsum = ops::add(va, vb);
    ops::store(out.data(), vsum);

    for (std::size_t i = 0; i < ops::width; ++i) REQUIRE(out[i] == TestType(i) + TestType(10));
}
