cmake_minimum_required(VERSION 3.20)
project(rt_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(rt STATIC src/rt.cpp)
target_include_directories(rt PUBLIC src)

add_executable(rt_test src/test_golden.cpp)
target_link_libraries(rt_test rt)

if(APPLE)
  target_link_libraries(rt "-framework Accelerate")
  target_compile_definitions(rt PRIVATE ACCELERATE_NEW_LAPACK)
endif()

add_executable(rt_bench src/bench.cpp)
target_link_libraries(rt_bench rt)

add_library(rt_c SHARED src/rt_c.cpp)
target_link_libraries(rt_c rt)
set_target_properties(rt_c PROPERTIES PUBLIC_HEADER src/rt_c.h)
