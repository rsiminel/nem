#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <latch>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nem {

    class ThreadPool {
    public:
        explicit ThreadPool(std::size_t n)
            : m_n_threads(std::max(std::size_t{1}, n)) {
            m_workers.reserve(m_n_threads);

            for (std::size_t i = 0; i < m_n_threads; ++i) {
                m_workers.emplace_back([this] { worker(); });
            }
        }

        ~ThreadPool() {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stop = true;
            }
            m_cv.notify_all();

            for (auto& w : m_workers) {
                w.join();
            }
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        std::size_t size() const noexcept {
            return m_n_threads;
        }

        void pfor(std::size_t begin, std::size_t end, const std::function<void(std::size_t)>& func) {
            if (begin >= end) {
                return;
            }

            std::size_t t = end - begin;
            std::size_t n_chunks = std::min(m_n_threads, t);
            std::size_t chunk_size = (t + n_chunks - 1) / n_chunks;

            std::latch done(static_cast<std::ptrdiff_t>(n_chunks));
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                for (std::size_t c = 0; c < n_chunks; ++c) {
                    std::size_t cbeg = begin + c * chunk_size;
                    std::size_t cend = std::min(end, cbeg + chunk_size);

                    if (cbeg >= cend) {
                        done.count_down();
                        continue;
                    }
                    m_tasks.push([&func, cbeg, cend, &done] {
                        for (std::size_t i = cbeg; i < cend; ++i) {
                            func(i);
                        }
                        done.count_down();
                    });
                }
            }
            m_cv.notify_all();
            done.wait();
        }

    private:
        void worker() {
            for(;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_cv.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

                    if (m_stop && m_tasks.empty()) {
                        return;
                    }
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }
                task();
            }
        }

    private:
        std::size_t m_n_threads {1};
        std::vector<std::thread> m_workers;
        std::queue<std::function<void()>> m_tasks;
        std::mutex m_mutex;
        std::condition_variable m_cv;
        bool m_stop = false;
    };

    inline void dispatch(ThreadPool* pool,
                         std::size_t begin,
                         std::size_t end,
                         const std::function<void(std::size_t)>& func) {

        if (pool != nullptr) {
            pool->pfor(begin, end, func);
            return;
        }

        for (std::size_t i = begin; i < end; ++i) {
            func(i);
        }

    }

    template<class T, class F>
    T parallel_sum(ThreadPool* pool, std::size_t begin, std::size_t end, F&& func) {
        if (begin >= end) {
            return T(0);
        }

        std::size_t n_threads = (pool != nullptr) ? pool->size() : std::size_t{1};
        std::size_t t = end - begin;
        std::size_t n_chunks = std::min(n_threads, t);
        std::size_t chunk_size = (t + n_chunks - 1) / n_chunks;

        std::vector<T> partials(n_chunks, T(0));
        dispatch(pool, 0, n_chunks, [&](std::size_t c) {
            std::size_t cbeg = begin + c * chunk_size;
            std::size_t cend = std::min(end, cbeg + chunk_size);
            T local = T(0);
            for (std::size_t i = cbeg; i < cend; ++i) {
                local += func(i);
            }
            partials[c] = local;
        });

        T total = T(0);
        for (T p : partials) {
            total += p;
        }
        return total;
    }
}
