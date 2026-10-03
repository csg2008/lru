# libFuzzer targets.
#
# These exist because the deserializers are untrusted-input boundaries — a
# snapshot can arrive corrupt, truncated, or hostile — and had no fuzz coverage
# at all. libFuzzer is part of clang, so this adds no third-party dependency;
# the target is only built when LRU_BUILD_FUZZERS=ON.

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "LRU_BUILD_FUZZERS requires clang (libFuzzer is a clang runtime)")
endif()

set(LRU_FUZZ_FLAGS -g -O1 -fno-omit-frame-pointer -fsanitize=fuzzer,address,undefined)

# The library is header-only; the fuzzer TU is the only translation unit.
set(LRU_FUZZ_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../unordered_dense/include)

function(lru_add_fuzzer target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${LRU_FUZZ_INCLUDES})
    target_compile_options(${target} PRIVATE ${LRU_FUZZ_FLAGS})
    target_link_options(${target} PRIVATE ${LRU_FUZZ_FLAGS})
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON)
endfunction()

lru_add_fuzzer(lru_fuzz_deserialize fuzz_deserialize.cpp)
lru_add_fuzzer(lru_fuzz_load fuzz_load.cpp)
