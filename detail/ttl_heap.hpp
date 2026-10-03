// strategy-agnostic TTL expiry index.
//
// Why this exists
// ---------------
// TTL used to be a per-strategy concern. `mm_lru` (and `sharded_mm_lru`) had an
// item-level expiry field plus this O(log n) min-heap, while `mm_fifo`, `mm_2q`,
// `mm_tiny_lfu` and `mm_wtiny_lfu` had NO item-level expiry at all — the field
// lived in the shared `cache_item` but nothing ever set it, so their TTL checks
// were dead code. Users of those strategies could only get TTL through the
// value-layer `ttl_entry<V>` wrapper, which the cache layer could sweep only by
// walking the ENTIRE cache under a read lock (`evict_expired_via_ttl_entry_scan`).
// That is O(n) per cleanup round, and it is why "TTL is O(log n)" was true for
// LRU and false for everything else.
//
// This header makes the index a plain, testable component instead of 40 lines
// duplicated inside a strategy. A strategy supplies two callbacks and gets the
// same O(log n) semantics and the same batch parameter as LRU:
//
//   * `probe(key, heap_expiry)` classifies the popped entry, and
//   * `erase(key)` performs the strategy's EXPIRY-AWARE removal — it must run the
//     expiration path (firing `on_expire`), not the capacity-eviction path
//     (which fires `on_evict`). Getting that wrong silently misreports TTL
//     expiry as capacity eviction and corrupts the operator-facing metrics.
//
// `Key` deliberately needs NO ordering: unlike a `(expiry, key)`-sorted index,
// nothing here compares keys — the min-heap orders by expiry alone and the
// strategy's own hash map answers "is this key still current?".

#ifndef LRU_DETAIL_TTL_HEAP_HPP
#define LRU_DETAIL_TTL_HEAP_HPP

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace lru::detail {

/// Min-heap of `(expiry_ns, key)` used to find expired items in O(log n).
///
/// Thread-safety: none of its own. Callers hold their strategy's write lock,
/// which is the existing contract for TTL cleanup (`mm_lru::evict_expired()`
/// documents the same requirement).
template <typename Key>
class ttl_heap {
public:
    /// Classification of a heap entry, supplied by the owning strategy.
    enum class probe_result {
        absent,  ///< key no longer in the map — discard the heap entry
        stale,   ///< key present but its expiry changed — discard this entry
        pinned,  ///< key present, current, but held by a live read_handle
        ready    ///< key present, current and unpinned — erase it
    };

    struct entry {
        std::uint64_t expiry_ns;
        Key key;
    };

    /// Comparator for std::push_heap/std::pop_heap (min-heap on expiry).
    struct min_on_expiry {
        bool operator()(const entry& a, const entry& b) const noexcept {
            return a.expiry_ns > b.expiry_ns;
        }
    };

    /// Record (or refresh) a key's expiry. `expiry_ns == 0` means "no TTL" and is
    /// ignored, matching the convention used across the library.
    ///
    /// Repeated updates of the same key push a new entry and leave the old one
    /// stale; it is discarded lazily by `sweep()` once its (old) expiry is
    /// reached. `needs_rebuild()` bounds how far that can grow.
    void push(const Key& key, std::uint64_t expiry_ns) {
        if (expiry_ns == 0) return;
        heap_.push_back(entry{expiry_ns, key});
        std::push_heap(heap_.begin(), heap_.end(), min_on_expiry{});
    }

    /// Evict expired entries.
    ///
    /// \param now_ns  current steady-clock value in nanoseconds
    /// \param budget  maximum number of evictions, or 0 for "no limit"
    /// \param probe   `probe_result probe(const Key&, std::uint64_t heap_expiry)`
    /// \param erase   `void erase(const Key&)` — the EXPIRY-AWARE removal
    /// \return number of entries erased
    ///
    /// A pinned entry is moved to a deferred queue and the sweep KEEPS GOING
    /// (P1-31 sub-fix A1). Because the heap is ordered by expiry, a pinned entry
    /// sits at the top, so the previous "push it back and break" behaviour let a
    /// single long-lived handle over the earliest-expiring key block every other
    /// expiry: the items behind it stayed resident indefinitely and each call
    /// paid a pop+push for nothing. The deferred queue is drained back into the
    /// heap at the end so the entry is retried once the handle is released.
    template <typename Probe, typename Erase>
    std::size_t sweep(std::uint64_t now_ns, std::size_t budget,
                      Probe&& probe, Erase&& erase) {
        std::size_t evicted = 0;
        // Safe to clear unconditionally: reinsert_deferred() below drains this
        // queue, so it is always empty on entry.
        deferred_.clear();
        while (!heap_.empty()) {
            if (budget != 0 && evicted >= budget) break;
            const auto& top = heap_.front();
            if (top.expiry_ns > now_ns) break;  // earliest expiry is future — done
            Key key = top.key;
            const std::uint64_t heap_expiry = top.expiry_ns;
            std::pop_heap(heap_.begin(), heap_.end(), min_on_expiry{});
            heap_.pop_back();

            switch (probe(key, heap_expiry)) {
                case probe_result::absent:
                case probe_result::stale:
                    continue;
                case probe_result::pinned:
                    deferred_.push_back(entry{heap_expiry, std::move(key)});
                    continue;
                case probe_result::ready:
                    erase(key);
                    ++evicted;
                    break;
            }
        }
        reinsert_deferred();
        return evicted;
    }

    /// True when the heap has accumulated far more entries than there are live
    /// items, i.e. most entries are stale and a rebuild would reclaim them.
    /// Amortized O(1) per push: the O(live) rebuild fires once per `multiplier`
    /// worth of pushes.
    bool needs_rebuild(std::size_t live_items, std::size_t multiplier) const noexcept {
        return heap_.size() > multiplier * live_items;
    }

    /// Drop everything and re-derive the index from the strategy's live items.
    ///
    /// \param push_live  invoked as `push_live(emit)` where
    ///        `emit(const Key&, std::uint64_t expiry_ns)` re-adds one live item.
    template <typename PushLive>
    void rebuild(PushLive&& push_live) {
        clear();
        auto emit = [this](const Key& key, std::uint64_t expiry_ns) {
            push(key, expiry_ns);
        };
        std::forward<PushLive>(push_live)(emit);
    }

    void clear() noexcept {
        heap_.clear();
        deferred_.clear();
    }

    std::size_t size() const noexcept { return heap_.size(); }
    bool empty() const noexcept { return heap_.empty(); }
    /// Entries deferred by the most recent sweep() because they were pinned.
    std::size_t deferred_count() const noexcept { return deferred_.size(); }

private:
    void reinsert_deferred() {
        for (auto& e : deferred_) {
            heap_.push_back(std::move(e));
            std::push_heap(heap_.begin(), heap_.end(), min_on_expiry{});
        }
        deferred_.clear();
    }

    std::vector<entry> heap_;
    std::vector<entry> deferred_;
};

}  // namespace lru::detail

namespace lru::detail {

/// CRTP mixin giving an eviction strategy a complete, O(log n), item-level TTL
/// facility.
///
/// before this, item-level TTL existed only in `mm_lru`.
/// The other strategies could obtain TTL solely from the value-layer
/// `ttl_entry<V>` wrapper, which the cache layer had to sweep by walking the
/// ENTIRE cache under a read lock. A strategy now derives from this mixin and
/// supplies four small answers; everything else — the ordered index, stale-entry
/// validation, the pinned-entry deferral, the growth-bounded rebuild and the
/// batch parameter — is shared:
///
///   * `ttl_probe_entry(key, heap_expiry) -> probe_result`
///         (absent / stale / pinned / ready)
///   * `ttl_erase_expired(key)`
///         MUST run the strategy's EXPIRATION path (firing `on_expire`), not its
///         capacity-eviction path (`on_evict`); otherwise TTL expiry is silently
///         accounted as capacity eviction (see the note in `ttl_heap`).
///   * `ttl_live_count() -> std::size_t`
///   * `ttl_for_each_live(emit)` — called as `emit(const Key&, std::uint64_t)`
///
/// New code should prefer this mixin over hand-rolling an index: `mm_lru`'s
/// original hand-rolled version and this one were diverging copies, which is
/// exactly the hazard P1-31 set out to remove.
template <typename Derived, typename Key>
class mm_ttl_index_mixin {
protected:
    using ttl_probe_result = ttl_heap<Key>::probe_result;

    /// Multiplier bounding index growth: a rebuild from the live set fires once
    /// the index holds more than this many entries per live item.
    static constexpr std::size_t kMaxTtlHeapMultiplier = 4;

    /// Record (or refresh) a key's expiry in the index.
    void ttl_index_push(const Key& key, std::uint64_t expiry_ns) {
        index_.push(key, expiry_ns);
        if (index_.needs_rebuild(derived().ttl_live_count(), kMaxTtlHeapMultiplier)) {
            ttl_index_rebuild();
        }
    }

    /// Discard everything and re-derive the index from the live items. Callers
    /// must hold the strategy's write lock.
    void ttl_index_rebuild() {
        index_.rebuild([this](auto&& emit) { derived().ttl_for_each_live(emit); });
    }

    /// Drop the index (used by paths that clear the whole strategy).
    void ttl_index_clear() noexcept { index_.clear(); }

    /// One sweep over the index, bounded by `budget` (0 = unlimited).
    std::size_t ttl_index_sweep(std::size_t budget) {
        const auto now_ns = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        return index_.sweep(
            now_ns, budget,
            [this](const Key& key, std::uint64_t heap_expiry) {
                return derived().ttl_probe_entry(key, heap_expiry);
            },
            [this](const Key& key) { derived().ttl_erase_expired(key); });
    }

    std::size_t ttl_index_size() const noexcept { return index_.size(); }

public:
    /// Evict every expired item (all expired entries are removed).
    std::size_t evict_expired() { return ttl_index_sweep(0); }

    /// Batched variant: evict at most `batch_size` expired items, so the
    /// non-blocking TTL cleaner never holds the write lock for long under heavy
    /// expiry load. `batch_size == 0` means "no limit".
    std::size_t evict_expired_n(std::size_t batch_size) {
        if (batch_size == 0) return evict_expired();
        return ttl_index_sweep(batch_size);
    }

private:
    Derived& derived() noexcept { return static_cast<Derived&>(*this); }

    ttl_heap<Key> index_;
};

}  // namespace lru::detail

#endif  // LRU_DETAIL_TTL_HEAP_HPP
