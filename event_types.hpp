// SPDX-License-Identifier: MIT
// Event Types — Shared definitions for event tracking
//
// Extracted from event_tracker.hpp so that tls_ring.hpp can reference
// event_type and event_record without creating a circular dependency.

#ifndef LRU_EVENT_TYPES_HPP
#define LRU_EVENT_TYPES_HPP

#include <cstdint>
#include <string_view>

namespace lru {

// ============================================================================
// Event Types
// ============================================================================

/// Types of lifecycle events that can be tracked.
enum class event_type : uint8_t {
    insert = 0,   // Item created/inserted
    promote = 1,  // Item promoted toward MRU
    demote = 2,   // Item demoted toward LRU
    evict = 3,    // Item evicted
    hit = 4,      // Item hit (read access)
};

/// Human-readable event type name.
inline constexpr std::string_view event_name(event_type et) {
    switch (et) {
        case event_type::insert:  return "insert";
        case event_type::promote: return "promote";
        case event_type::demote:  return "demote";
        case event_type::evict:   return "evict";
        case event_type::hit:     return "hit";
    }
    return "unknown";
}

// ============================================================================
// Event Record
// ============================================================================

/// A single tracked event with key and timestamp.
// H-9 fix: alignas(32) to prevent split loads/stores when
// store_entry_atomic() uses std::atomic_ref on individual fields.
// Previously 25 bytes (8+8+1+1), crossing cache line boundaries
// and causing atomic_ref to generate split accesses that are
// several times slower than naturally-aligned atomics.
struct alignas(32) event_record {
    uint64_t key_hash;         // Hash of the item key (space-efficient)
    uint64_t timestamp_ms;     // When the event occurred (ms since steady_clock epoch; R6: monotonic)
    event_type type;           // What happened
    uint8_t queue_id = 0;      // Queue ID (for multi-queue strategies)
};

} // namespace lru

#endif // LRU_EVENT_TYPES_HPP
