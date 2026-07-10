#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fastnem/core/matrix.hpp>

TEMPLATE_TEST_CASE("Matrix",
                    "[matrix]", float, double) {

    nem::Matrix<TestType> m(3, 4);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            m(r, c) = static_cast<TestType>(r * 10 + c);
        }
    }
    REQUIRE(m(0, 0) == TestType(0));
    REQUIRE(m(2, 3) == TestType(23));
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 4);
}

