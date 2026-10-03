// SPDX-License-Identifier: MIT
// ThreadSanitizer annotation helpers.
//
// A few places in this library use lock-free/seqlock patterns that are
// *logically* race-free (the reader validates a snapshot and discards it when
// a concurrent writer invalidated it) but are formally data races on
// non-atomic objects: the C++ memory model has no notion of "I will check
// afterwards", so any concurrent read/write pair on a non-atomic object is UB,
// and TSan reports it.
//
// CacheLib silences exactly these reads with the TSan annotate interface (see
// DList.h's updateTime accessor, which this library already mirrors). This
// header centralises the declarations so every site uses the same definition
// and the library never has to depend on the annotation functions being
// declared by a system header.
//
// Contract: only wrap reads that are *deliberately* tolerant of a concurrent
// writer — i.e. the result is validated (seqlock re-check, hazard-pointer
// protection, or refcount pinning) before it is relied upon. Never use this to
// paper over a missing lock.

#ifndef LRU_DETAIL_TSAN_ANNOTATIONS_HPP
#define LRU_DETAIL_TSAN_ANNOTATIONS_HPP

#if defined(__SANITIZE_THREAD__) || (defined(__has_feature) && __has_feature(thread_sanitizer))
#define LRU_HAS_TSAN 1
#if __has_include(<sanitizer/tsan_interface_ann.h>)
#include <sanitizer/tsan_interface_ann.h>
#else
// Clang no longer ships tsan_interface_ann.h, but the TSan runtime still
// exports the annotation entry points. Declare the ones used here so the
// build links; when TSan is disabled this whole block is compiled out.
extern "C" void AnnotateIgnoreReadsBegin(const char* file, int line);
extern "C" void AnnotateIgnoreReadsEnd(const char* file, int line);
extern "C" void AnnotateIgnoreWritesBegin(const char* file, int line);
extern "C" void AnnotateIgnoreWritesEnd(const char* file, int line);
#endif
#endif

#if defined(LRU_HAS_TSAN)
#define LRU_TSAN_IGNORE_READS_BEGIN() AnnotateIgnoreReadsBegin(__FILE__, __LINE__)
#define LRU_TSAN_IGNORE_READS_END()   AnnotateIgnoreReadsEnd(__FILE__, __LINE__)
#define LRU_TSAN_IGNORE_WRITES_BEGIN() AnnotateIgnoreWritesBegin(__FILE__, __LINE__)
#define LRU_TSAN_IGNORE_WRITES_END()   AnnotateIgnoreWritesEnd(__FILE__, __LINE__)
#else
#define LRU_TSAN_IGNORE_READS_BEGIN() ((void)0)
#define LRU_TSAN_IGNORE_READS_END()   ((void)0)
#define LRU_TSAN_IGNORE_WRITES_BEGIN() ((void)0)
#define LRU_TSAN_IGNORE_WRITES_END()   ((void)0)
#endif

#endif // LRU_DETAIL_TSAN_ANNOTATIONS_HPP
