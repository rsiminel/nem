add_executable(fastnem_test
    core/aligned_vector.cpp
    core/matrix.cpp
    core/csr_graph.cpp
    core/simd.cpp
)

target_link_libraries(fastnem_test PRIVATE fastnem Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(fastnem_test)
