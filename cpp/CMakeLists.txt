cmake_minimum_required(VERSION 3.20)
project(rt_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# rt_c is a SHARED library that links the rt/relql/csc STATIC libraries. On ELF
# targets (Linux) every object pulled into a .so must be position independent or
# the link dies with "relocation R_X86_64_PC32 against symbol ... can not be
# used when making a shared object". macOS is PIC by default, which is why this
# was never noticed here.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Backends: CPU is always built (Accelerate on Apple, portable SIMD kernels
# elsewhere). RT_METAL adds the Metal/MPS GPU backend (Apple only, default ON).
# RT_CUDA adds the cuBLAS/CUDA GPU backend (requires the CUDA toolkit).
option(RT_METAL "Build the Metal/MPS backend (Apple only)" ON)
option(RT_CUDA "Build the CUDA backend" OFF)
option(RT_WARNINGS "Enable -Wall -Wextra (never -Werror)" ON)
option(RT_COVERAGE "Build with coverage instrumentation" OFF)
option(RT_ASSERTS "Keep assert() live in optimized builds" ON)

# --- toolchain-wide flags ---------------------------------------------------
# NOTE: CMAKE_CXX_FLAGS_RELEASE is deliberately NOT overwritten here. CMake's
# own default for GCC/Clang is already "-O3 -DNDEBUG"; hardcoding it clobbered
# anything the caller passed on the command line.
add_library(rt_flags INTERFACE)

if(RT_WARNINGS)
  target_compile_options(rt_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
  # GCC flags the codebase's dense one-line `if (x) a; b;` style in rt_c.cpp as
  # misleading indentation. All six sites were reviewed and are correct; the
  # warning is noise here. Never -Werror: new compilers add new warnings and
  # that would make CI fail on unrelated toolchain bumps.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(rt_flags INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-misleading-indentation>)
  endif()
endif()

# The safetensors header scanner validates structure with assert(). Release adds
# -DNDEBUG, which would turn a malformed checkpoint from a clean abort into
# undefined behaviour, so re-enable assertions unless the user opts out.
if(RT_ASSERTS)
  target_compile_options(rt_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
endif()

if(RT_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")   # Clang and AppleClang
    set(RT_COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(RT_COVERAGE_FLAGS --coverage)
  else()
    message(FATAL_ERROR
      "RT_COVERAGE supports GCC and Clang only (got ${CMAKE_CXX_COMPILER_ID})")
  endif()
  # -O0 -g -fno-inline so line mapping is usable; coverage of an -O3 build is
  # misleading.
  target_compile_options(rt_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${RT_COVERAGE_FLAGS};-O0;-g;-fno-inline>)
  target_link_options(rt_flags INTERFACE ${RT_COVERAGE_FLAGS})
  message(STATUS "RT_COVERAGE: ON (${RT_COVERAGE_FLAGS})")
endif()

# <thread>/<mutex>/<condition_variable> in rt.cpp need -pthread on Linux.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- libraries --------------------------------------------------------------
add_library(rt STATIC src/rt.cpp src/rt_train.cpp)
target_include_directories(rt PUBLIC src)
target_link_libraries(rt PUBLIC rt_flags Threads::Threads)

if(APPLE)
  target_link_libraries(rt PUBLIC "-framework Accelerate")
  target_compile_definitions(rt PRIVATE ACCELERATE_NEW_LAPACK)
  if(RT_METAL)
    target_sources(rt PRIVATE src/rt_metal.mm src/rt_train_metal.mm
                              src/rt_full_train_metal.mm)
    target_compile_definitions(rt PUBLIC RT_METAL)
    set_source_files_properties(src/rt_metal.mm src/rt_train_metal.mm
                                src/rt_full_train_metal.mm PROPERTIES
                                COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(rt PUBLIC "-framework Metal"
                                    "-framework MetalPerformanceShaders"
                                    "-framework Foundation")
  endif()
endif()

if(RT_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(rt PRIVATE src/rt_cuda.cu)
  target_compile_definitions(rt PUBLIC RT_CUDA)
  target_link_libraries(rt PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(rt PROPERTIES CUDA_STANDARD 17)
endif()

# Shared RelQL front end: lexer + recursive-descent parser, plus the
# schema-aware semantic pass (validation, task-type inference) that each
# binding used to reimplement in its own language.
add_library(relql STATIC src/relql.cpp src/schema.cpp src/analyze.cpp
                        src/plan.cpp src/json.cpp)
target_include_directories(relql PUBLIC src)
target_link_libraries(relql PUBLIC rt_flags)

# Shared CSC adjacency (time-bounded "latest <= anchor" children query).
add_library(csc STATIC src/csc.cpp)

# Array-backed context assembly: returns the ordered node ids a
# context emits, so a binding materializes only those rows.
add_library(relgraph STATIC src/graph.cpp)
target_include_directories(relgraph PUBLIC src)
target_include_directories(csc PUBLIC src)
target_link_libraries(csc PUBLIC rt_flags)

add_library(rt_c SHARED src/rt_c.cpp src/relql_c.cpp src/relql.cpp
                        src/schema.cpp src/analyze.cpp src/plan.cpp src/json.cpp
                        src/csc_c.cpp src/csc.cpp
                        src/graph.cpp src/graph_c.cpp)
target_link_libraries(rt_c PRIVATE rt)
target_include_directories(rt_c PUBLIC src)
set_target_properties(rt_c PROPERTIES PUBLIC_HEADER src/rt_c.h)

# --- executables ------------------------------------------------------------
add_executable(rt_test src/test_golden.cpp)
target_link_libraries(rt_test rt)

add_executable(rt_bench src/bench.cpp)
target_link_libraries(rt_bench rt)

# Concurrent-request (multiuser serving shape) benchmark.
add_executable(rt_bench_mu src/bench_multiuser.cpp)
target_link_libraries(rt_bench_mu rt)

add_executable(rt_train_test src/test_train.cpp)
target_link_libraries(rt_train_test rt)

# Checkpoint quantizer: fp32/bf16 safetensors -> int8 per-row (Q8_0-style).
add_executable(rt_quantize src/quantize.cpp)
target_link_libraries(rt_quantize rt)

add_executable(relql_test src/test_relql.cpp)
target_link_libraries(relql_test relql)

# The semantic pass: validation + schema-aware task typing, shared by every
# binding, so its behaviour is pinned here rather than per language.
add_executable(analyze_test src/test_analyze.cpp)
target_link_libraries(analyze_test relql)

# csc_c.cpp is compiled in directly (rather than linking rt_c) so the test
# exercises the C ABI without dragging in the whole engine shared library.
add_executable(graph_test src/test_graph.cpp)
target_link_libraries(graph_test relgraph)

add_executable(csc_test src/test_csc.cpp src/csc_c.cpp)
target_link_libraries(csc_test csc)

# --- tests ------------------------------------------------------------------
# Contract: `ctest --test-dir <builddir> --output-on-failure` runs the fast,
# hermetic suite (no network, no checkpoint, a couple of seconds total).
include(CTest)

if(BUILD_TESTING)
  # Self-contained: randomized CSC adjacency batteries with fixed seeds.
  add_test(NAME csc COMMAND csc_test)
  add_test(NAME analyze COMMAND analyze_test)
  add_test(NAME graph COMMAND graph_test)

  # Needs the RelQL corpus, which lives in the python package. Pass it as an
  # absolute path so the test is independent of the working directory.
  add_test(NAME relql COMMAND relql_test
           "${CMAKE_CURRENT_SOURCE_DIR}/../python/tests/data/examples.relql")

  # Head fine-tuning. The zero-shot reparameterization check runs everywhere;
  # the Metal sections self-skip (exit 0, "SKIP: Metal unavailable") when no MPS
  # device is present, so this is safe on Linux CI.
  add_test(NAME train COMMAND rt_train_test)

  set_tests_properties(csc relql train PROPERTIES
                       TIMEOUT 300
                       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

  # Golden forward-pass test. NOT part of the default suite: it needs both the
  # dumped demo batch (cpp/testdata/*.bin, which is .gitignored -- regenerate
  # with cpp/tools/dump_golden.py) and a real .safetensors checkpoint. Enable
  # with -DRT_TEST_CHECKPOINT=/path/to/model.safetensors and run
  # `ctest -L integration`; without it the test is registered but DISABLED.
  set(RT_TEST_CHECKPOINT "" CACHE FILEPATH
      "Checkpoint for the rt_golden integration test")
  set(RT_TESTDATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/testdata" CACHE PATH
      "Directory holding the dumped golden batch (*.bin)")
  add_test(NAME rt_golden
           COMMAND rt_test "${RT_TESTDATA_DIR}" "${RT_TEST_CHECKPOINT}")
  set_tests_properties(rt_golden PROPERTIES
                       LABELS "integration"
                       TIMEOUT 900
                       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  if(NOT RT_TEST_CHECKPOINT OR NOT EXISTS "${RT_TESTDATA_DIR}/node_idxs.bin")
    set_tests_properties(rt_golden PROPERTIES DISABLED TRUE)
  endif()

  # Same gate on the long-context golden (B=2, S=70). The batch above is B=5
  # S=16, which is below the sequence length where crowded attention groups
  # start to exist at all -- a grouping or masking defect that only bites past
  # S~36 passes it green. --tol 6 because the stored reference is the PyTorch
  # model in fp32, whose own drift at S=70 is ~4x its drift at S=16.
  add_test(NAME rt_golden_large
           COMMAND rt_test "${RT_TESTDATA_DIR}/large" "${RT_TEST_CHECKPOINT}"
                   --seq 70 --tol 6)
  set_tests_properties(rt_golden_large PROPERTIES
                       LABELS "integration"
                       TIMEOUT 900
                       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  if(NOT RT_TEST_CHECKPOINT OR NOT EXISTS "${RT_TESTDATA_DIR}/large/node_idxs.bin")
    set_tests_properties(rt_golden_large PROPERTIES DISABLED TRUE)
  endif()
endif()
