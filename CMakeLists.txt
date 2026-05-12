cmake_minimum_required(VERSION 3.20)

project(gpu-benchmarks LANGUAGES CXX)

set(BENCHMARK_BACKEND "CUDA" CACHE STRING "GPU backend to build (CUDA or HIP)")
set_property(CACHE BENCHMARK_BACKEND PROPERTY STRINGS CUDA HIP)

set(CHARM_ROOT "" CACHE PATH "Path to Charm++ installation (expects bin/charmc)")
set(KOKKOS_ROOT "" CACHE PATH "Path to Kokkos installation")

if(CHARM_ROOT STREQUAL "")
  message(FATAL_ERROR "CHARM_ROOT is required (path containing bin/charmc).")
endif()

if(KOKKOS_ROOT STREQUAL "")
  message(FATAL_ERROR "KOKKOS_ROOT is required.")
endif()

if(NOT BENCHMARK_BACKEND STREQUAL "CUDA" AND NOT BENCHMARK_BACKEND STREQUAL "HIP")
  message(FATAL_ERROR "BENCHMARK_BACKEND must be CUDA or HIP.")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Charm.cmake)

add_subdirectory(benchmarks/hapi-kernel-launch)
