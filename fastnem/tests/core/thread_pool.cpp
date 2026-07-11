#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <vector>

#include <fastnem/core/thread_pool.hpp>

TEST_CASE("ThreadPool", "[thread_pool]") {
    nem::ThreadPool pool(4);
    constexpr std::size_t n = 1000;
    std::vector<std::atomic<int>> hits(n);
    for (auto& h : hits) h = 0;

    pool.pfor(0, n, [&hits](std::size_t i) { hits[i].fetch_add(1); });

    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(hits[i].load() == 1);
    }
}

TEST_CASE("ThreadPool dispatch", "[thread_pool]") {
    nem::ThreadPool pool(4);
    constexpr std::size_t n = 1000;
    std::vector<std::atomic<int>> hits(n);
    for (auto& h : hits) h = 0;

    nem::dispatch(&pool, 0, n, [&hits](std::size_t i) { hits[i].fetch_add(1); });

    for (std::size_t i = 0; i < n; ++i) REQUIRE(hits[i].load() == 1);
}

