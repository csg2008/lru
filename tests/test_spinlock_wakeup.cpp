// SPDX-License-Identifier: MIT
// Regression tests for the per-bucket shared_spinlock wake protocol.
//
// unlock() previously skipped the wake whenever the releasing writer saw
// no waiting-writer flag and no reader-count bit in the state word it captured.
// A parked reader is invisible in that word — both reader slow paths undo the
// reader-count increment before parking — so the guard stranded every reader
// that queued behind a writer. Because shared_spinlock is the per-bucket
// read/write lock, that hang affected every hash lookup under a long writer
// critical section.
//
// These tests drive the exact interleaving deterministically: a writer holds
// the lock while a second thread blocks in lock_shared(), then releases. The
// blocked reader must complete; the join is bounded so a regression fails
// instead of hanging the suite.

#include "detail/concurrent_hash_table.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using lru::detail::shared_spinlock;

}  // namespace

// The core reproducer: a reader that parks behind a writer must be released by
// the writer's unlock().
TEST(SpinlockWakeup, WriterUnlockReleasesParkedReader) {
    shared_spinlock lock;

    for (int attempt = 0; attempt < 50; ++attempt) {
        std::atomic<bool> reader_done{false};

        lock.lock();  // writer holds the bucket

        std::thread reader([&] {
            lock.lock_shared();
            lock.unlock_shared();
            reader_done.store(true, std::memory_order_release);
        });

        // Give the reader time to exhaust its spin budget and park on the
        // native wait primitive (it observes kWriterFlag and sleeps).
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        lock.unlock();  // must wake the parked reader

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!reader_done.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        const bool finished = reader_done.load(std::memory_order_acquire);
        if (!finished) {
            // Detach so the process can report the failure and exit; the
            // stranded thread would otherwise block the join forever.
            std::fprintf(stderr,
                         "[SpinlockWakeup] attempt %d: reader still blocked "
                         "3s after writer unlocked\n",
                         attempt);
            reader.detach();
            FAIL() << "parked reader was not released by unlock()";
        }
        reader.join();
    }
}

// Several readers park behind one writer; the single unlock must release them
// all (do_wake_all, not do_wake_one).
TEST(SpinlockWakeup, WriterUnlockReleasesAllParkedReaders) {
    constexpr int kReaders = 8;
    shared_spinlock lock;

    for (int attempt = 0; attempt < 20; ++attempt) {
        std::vector<std::thread> readers;
        std::atomic<int> done{0};

        lock.lock();

        readers.reserve(kReaders);
        for (int i = 0; i < kReaders; ++i) {
            readers.emplace_back([&] {
                lock.lock_shared();
                lock.unlock_shared();
                done.fetch_add(1, std::memory_order_release);
            });
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        lock.unlock();

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (done.load(std::memory_order_acquire) < kReaders &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        const int completed = done.load(std::memory_order_acquire);
        if (completed < kReaders) {
            for (auto& t : readers) {
                if (t.joinable()) t.detach();
            }
            FAIL() << "only " << completed << " of " << kReaders
                   << " parked readers were released by unlock()";
        }
        for (auto& t : readers) t.join();
    }
}

// Writer-wait wake path: a reader holds the lock while a writer queues. The
// reader's unlock_shared() must wake the queued writer (do_wake_one).
TEST(SpinlockWakeup, LastReaderWakesQueuedWriter) {
    shared_spinlock lock;

    for (int attempt = 0; attempt < 50; ++attempt) {
        std::atomic<bool> writer_done{false};

        lock.lock_shared();

        std::thread writer([&] {
            lock.lock();
            lock.unlock();
            writer_done.store(true, std::memory_order_release);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        lock.unlock_shared();  // must wake the queued writer

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!writer_done.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        const bool finished = writer_done.load(std::memory_order_acquire);
        if (!finished) {
            writer.detach();
            FAIL() << "queued writer was not woken by the last reader";
        }
        writer.join();
    }
}
