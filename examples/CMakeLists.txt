# Examples CMakeLists.txt

lru_add_internal_executable(basic_example basic_example.cpp)
lru_add_internal_executable(threadsafe_example threadsafe_example.cpp)
lru_add_internal_executable(callbacks_example callbacks_example.cpp)
lru_add_internal_executable(ttl_example ttl_example.cpp)

# Recommended production configuration for read-heavy, high-concurrency workloads.
lru_add_internal_executable(production_example production_example.cpp)

# Minimal reproduction for destructor crash debugging.
lru_add_internal_executable(minimal_repro minimal_repro.cpp)

# Operator-facing diagnostic dump: drives a cache and periodically prints
# diagnostics_text() plus the eviction breakdown by reason, so it can be copied
# into a SIGUSR1 handler or an admin endpoint.
lru_add_internal_executable(lru_diag lru_diag.cpp)
