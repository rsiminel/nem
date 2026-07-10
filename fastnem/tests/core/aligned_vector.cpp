#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <utility>

#include <fastnem/core/aligned_vector.hpp>


TEMPLATE_TEST_CASE("AlignedVector init", "[aligned_vector]", float, double) {
    nem::AlignedVector<TestType> v(16);
    REQUIRE(v.size() == 16);
    REQUIRE(reinterpret_cast<std::uintptr_t>(v.data()) % 32 == 0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE(v[i] == TestType(0));
    }
}
