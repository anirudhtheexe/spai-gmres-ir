# Compatibility entry point for standalone CUDA builds.
cmake_minimum_required(VERSION 3.18)
project(spai_cuda LANGUAGES CXX CUDA)
include(CTest)
set(BSPAI_ENABLE_CUDA ON CACHE BOOL "Build CUDA solvers" FORCE)
# Run original CPU wide-long-double oracle suites separately with GCC/Clang.
set(BSPAI_BUILD_CPU_TESTS OFF CACHE BOOL "Build CPU regression tests")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." solver)
