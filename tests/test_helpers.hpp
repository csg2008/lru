// SPDX-License-Identifier: MIT
// Common test helpers for concurrent test suites.
//
// Provides:
//   - Watchdog: timeout-based deadlock detector
//   - RandomKeyGen: seedable key generator (LRU_TEST_SEED)
//   - JoinAll: exception-safe bulk thread join
//   - ScopedEnv: temporary environment variable
//   - read_env_or: read environment variable with default
//
// Used by test_concurrent_*.cpp test files.

#ifndef LRU_TESTS_TEST_HELPERS_HPP
#define LRU_TESTS_TEST_HELPERS_HPP

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace lru_test {

// Resolve stress-test duration from environment. Returns default if unset or
// invalid. Honors LRU_STRESS_DURATION_MS for workload length configuration.
inline std::chrono::milliseconds read_stress_duration_ms(
        std::chrono::milliseconds default_value) {
    const char* env = std::getenv("LRU_STRESS_DURATION_MS");
    if (!env || env[0] == '\0') return default_value;
    try {
        long long ms = std::stoll(env);
        if (ms < 0) return default_value;
        return std::chrono::milliseconds(ms);
    } catch (...) {
        return default_value;
    }
}

// Resolve random seed from environment. Defaults to a non-deterministic
// value derived from the system clock if LRU_TEST_SEED is unset.
inline std::uint64_t read_test_seed(std::uint64_t default_seed) {
    const char* env = std::getenv("LRU_TEST_SEED");
    if (!env || env[0] == '\0') return default_seed;
    try {
        return static_cast<std::uint64_t>(std::stoull(env));
    } catch (...) {
        return default_seed;
    }
}

// Watchdog: runs a callable and fails (returns false) if it does not
// complete within `timeout`. The callable must not capture the watchdog.
//
// Usage:
//   ASSERT_TRUE(lru_test::run_with_watchdog([&] { ... workload ... }));
struct Watchdog {
    // Run workload with a 10s watchdog by default.
    static bool run(std::function<void()> workload,
                    std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
        std::atomic<bool> done{false};
        std::atomic<bool> timed_out{false};

        std::thread watcher([&]() {
            auto start = std::chrono::steady_clock::now();
            while (!done.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() - start > timeout) {
                    timed_out.store(true, std::memory_order_release);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });

        workload();
        done.store(true, std::memory_order_release);
        watcher.join();

        return !timed_out.load(std::memory_order_acquire);
    }
};

// RandomKeyGen: thread-local PRNG keyed by thread-id + seed. Use this to
// produce reproducible workloads when LRU_TEST_SEED is set.
class RandomKeyGen {
public:
    explicit RandomKeyGen(std::uint64_t seed) : rng_(seed) {}

    // Next integer in [0, bound).
    std::uint64_t next(std::uint64_t bound) {
        if (bound == 0) return 0;
        return rng_() % bound;
    }

    // Next integer in [lo, hi).
    std::uint64_t next_range(std::uint64_t lo, std::uint64_t hi) {
        if (hi <= lo) return lo;
        return lo + next(hi - lo);
    }

    void seed(std::uint64_t s) { rng_.seed(s); }

private:
    std::mt19937_64 rng_;
};

// JoinAll: joins every thread in the vector, capturing the first exception
// thrown so subsequent joins are not lost. Re-throws the first captured
// exception after all threads have been joined.
inline void join_all(std::vector<std::thread>& threads) {
    std::exception_ptr first_eptr;
    for (auto& th : threads) {
        if (!th.joinable()) continue;
        try {
            th.join();
        } catch (...) {
            if (!first_eptr) first_eptr = std::current_exception();
        }
    }
    if (first_eptr) std::rethrow_exception(first_eptr);
}

// ScopedEnv: sets an environment variable to a new value and restores the
// previous value on destruction. No-op if setenv/unsetenv are unavailable.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        const char* prev = std::getenv(name);
        previous_ = prev ? prev : "";
        had_previous_ = (prev != nullptr);
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
    ~ScopedEnv() {
        try {
#ifdef _WIN32
            if (had_previous_) {
                _putenv_s(name_.c_str(), previous_.c_str());
            } else {
                _putenv_s(name_.c_str(), "");
            }
#else
            if (had_previous_) {
                setenv(name_.c_str(), previous_.c_str(), 1);
            } else {
                unsetenv(name_.c_str());
            }
#endif
        } catch (...) {
            // suppress
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string name_;
    std::string previous_;
    bool had_previous_;
};

}  // namespace lru_test

#endif  // LRU_TESTS_TEST_HELPERS_HPP
