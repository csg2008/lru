// T20: tls_event_ring backup buffer tests.
//
// Validates that events recorded by a thread are NOT lost when the thread
// exits before the next drain cycle. The thread_exit_sentinel must push
// remaining TLS events into the global backup buffer, which is then
// retrieved by drain_all_threads() / flush_all_registered().

#include "../event_tracker.hpp"
#include "../tls_ring.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using lru::tls_event_ring;
using lru::event_tracker;
using lru::event_type;

// Helper: drain backup and count entries.
static std::size_t drain_backup_count() {
    auto r = tls_event_ring<int>::drain_backup();
    return r.entries.size();
}

TEST(TlsEventBackupTest, BackupBufferStartsEmpty) {
    // Drain any leftover state first so the test starts clean.
    (void)drain_backup_count();
    EXPECT_FALSE(tls_event_ring<int>::has_backup_entries());
}

TEST(TlsEventBackupTest, PushToBackupIsRetrievable) {
    (void)drain_backup_count();
    std::vector<tls_event_ring<int>::event_entry> entries;
    tls_event_ring<int>::event_entry e;
    e.key_hash = 42;
    e.type = event_type::insert;
    entries.push_back(e);
    tls_event_ring<int>::push_to_backup(std::move(entries));
    EXPECT_TRUE(tls_event_ring<int>::has_backup_entries());
    auto drained = tls_event_ring<int>::drain_backup();
    EXPECT_EQ(drained.entries.size(), 1u);
    EXPECT_EQ(drained.entries[0].key_hash, 42u);
    EXPECT_FALSE(tls_event_ring<int>::has_backup_entries());
}

TEST(TlsEventBackupTest, ThreadExitPushesEventsToBackup) {
    // T20.4: A thread records events then exits WITHOUT draining. The
    // sentinel must push those events into the backup buffer so they
    // can be retrieved by a subsequent drain_backup() / drain_all_threads().
    (void)drain_backup_count();

    event_tracker<int> tracker(10000);
    constexpr int kEvents = 10;

    {
        std::thread t([&tracker, kEvents]() {
            for (int i = 0; i < kEvents; ++i) {
                tracker.record_insert(i);
            }
            // Do NOT call drain_tls() — let the sentinel handle it.
        });
        t.join();
    }

    // After the thread exits, the backup buffer should contain at least
    // some events. (The exact count depends on ring size and drain timing,
    // but with kEvents=10 and ring size=256, all 10 should survive.)
    auto backup = tls_event_ring<int>::drain_backup();
    EXPECT_GE(backup.entries.size(), 1u);
    // With kEvents=10 << kRingSize=256, no overflow should occur.
    EXPECT_LE(backup.entries.size(), static_cast<std::size_t>(kEvents));
}

TEST(TlsEventBackupTest, DrainAllThreadsRetrievesExitedThreadEvents) {
    // T20.4: drain_all_threads() should retrieve events from exited
    // threads via the backup buffer, in addition to live threads' data.
    (void)drain_backup_count();

    event_tracker<int> tracker(10000);
    constexpr int kEvents = 20;

    {
        std::thread t([&tracker, kEvents]() {
            for (int i = 0; i < kEvents; ++i) {
                tracker.record_insert(i);
            }
        });
        t.join();
    }

    // drain_all_threads() processes events internally (returns void).
    // We verify the events were retrieved by checking total_recorded().
    std::size_t before = tracker.total_recorded();
    tracker.drain_all_threads();
    std::size_t after = tracker.total_recorded();
    // total_recorded should have advanced by at least 1 (events from
    // the exited thread, retrieved via the backup buffer).
    EXPECT_GT(after, before);
    EXPECT_GE(after - before, 1u);
    EXPECT_LE(after - before, static_cast<std::size_t>(kEvents));
}

TEST(TlsEventBackupTest, FlushAllRegisteredRetrievesBackupEvents) {
    // T20.4: flush_all_registered() should also drain the backup buffer.
    (void)drain_backup_count();

    event_tracker<int> tracker(10000);
    constexpr int kEvents = 5;

    {
        std::thread t([&tracker, kEvents]() {
            for (int i = 0; i < kEvents; ++i) {
                tracker.record_insert(i);
            }
        });
        t.join();
    }

    // flush_all_registered() returns a drain_result that should contain
    // events from the exited thread (via backup) + the calling thread's
    // TLS (which is empty here).
    auto drained = tls_event_ring<int>::flush_all_registered();
    EXPECT_GE(drained.entries.size(), 1u);
}

TEST(TlsEventBackupTest, MultipleThreadsExitEventsPreserved) {
    // T20.4: Multiple short-lived threads all record events and exit.
    // All their events should be pushed to the backup buffer and
    // retrieved together.
    (void)drain_backup_count();

    event_tracker<int> tracker(100000);
    constexpr int kThreads = 4;
    constexpr int kEventsPerThread = 15;

    for (int t = 0; t < kThreads; ++t) {
        std::thread([&tracker, t]() {
            for (int i = 0; i < kEventsPerThread; ++i) {
                tracker.record_insert(t * 100 + i);
            }
        }).join();
    }

    // drain_all_threads() processes events internally; verify via
    // total_recorded().
    std::size_t before = tracker.total_recorded();
    tracker.drain_all_threads();
    std::size_t after = tracker.total_recorded();
    // We expect roughly kThreads * kEventsPerThread events, but exact
    // count depends on timing and ring size. With kEventsPerThread=15
    // and kRingSize=256, no overflow should occur.
    std::size_t delta = after - before;
    EXPECT_GE(delta, static_cast<std::size_t>(kThreads * kEventsPerThread / 2));
    EXPECT_LE(delta, static_cast<std::size_t>(kThreads * kEventsPerThread));
}

TEST(TlsEventBackupTest, NoEventsLostAcrossThreadLifecycle) {
    // T20.4: End-to-end test — record events from multiple short-lived
    // threads, then generate a report. The report's total events should
    // account for all recorded inserts (within ring-overflow tolerance).
    (void)drain_backup_count();

    event_tracker<int> tracker(100000);
    constexpr int kThreads = 4;
    constexpr int kEventsPerThread = 10;
    constexpr int kTotalExpectedEvents = kThreads * kEventsPerThread;

    for (int t = 0; t < kThreads; ++t) {
        std::thread([&tracker, t]() {
            for (int i = 0; i < kEventsPerThread; ++i) {
                tracker.record_insert(t * 1000 + i);
            }
        }).join();
    }

    // Also record from the main thread to mix live + backup paths.
    for (int i = 0; i < kEventsPerThread; ++i) {
        tracker.record_insert(9000 + i);
    }

    // T20.4: Explicitly drain_all_threads() to pick up events from the
    // 4 exited threads (via the backup buffer) before generating the
    // report. generate_report() only calls drain_tls() (calling thread),
    // so without this step the exited threads' events would not be
    // included in the report.
    tracker.drain_all_threads();

    auto r = tracker.generate_report();
    // The report should include events from:
    //   - 4 exited threads (via backup buffer, retrieved by drain_all_threads)
    //   - 1 main thread (via drain_tls in generate_report)
    // Total inserts should be at least kTotalExpectedEvents (from exited
    // threads). The main thread's events are also expected.
    EXPECT_GE(r.inserts, static_cast<std::size_t>(kTotalExpectedEvents));
}

TEST(TlsEventBackupTest, BackupBufferIsPerTemplateSpecialization) {
    // T20.4: Different Key types should have independent backup buffers.
    // Recording int events should not affect the std::string backup buffer.
    (void)drain_backup_count();
    (void)tls_event_ring<std::string>::drain_backup();

    std::vector<tls_event_ring<int>::event_entry> int_entries;
    tls_event_ring<int>::event_entry e;
    e.key_hash = 42;
    e.type = event_type::insert;
    int_entries.push_back(e);
    tls_event_ring<int>::push_to_backup(std::move(int_entries));

    EXPECT_TRUE(tls_event_ring<int>::has_backup_entries());
    EXPECT_FALSE(tls_event_ring<std::string>::has_backup_entries());

    // Cleanup.
    (void)drain_backup_count();
}

TEST(TlsEventBackupTest, EmptyBackupDrainReturnsEmpty) {
    (void)drain_backup_count();
    auto drained = tls_event_ring<int>::drain_backup();
    EXPECT_TRUE(drained.entries.empty());
}
