set(BENCHMARK_NAME hapi_kernel_launch)
set(BENCHMARK_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/hapi_kernel_launch.C)
set(BENCHMARK_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/hapi_kernel_launch.ci)
set(KOKKOS_INSTALL_DIR "${KOKKOS_ROOT}/install")
set(KOKKOS_INCLUDE_DIR "${KOKKOS_INSTALL_DIR}/include")
set(KOKKOS_LIBRARY_DIR "${KOKKOS_INSTALL_DIR}/lib64")

if(NOT EXISTS "${KOKKOS_INCLUDE_DIR}")
  message(FATAL_ERROR "Kokkos include directory not found: ${KOKKOS_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${KOKKOS_LIBRARY_DIR}")
  message(FATAL_ERROR "Kokkos library directory not found: ${KOKKOS_LIBRARY_DIR}")
endif()

option(HAPI_KERNEL_LAUNCH_ENABLE_SAMPLING "Collect and print per-chare launch gap samples" OFF)

add_charm_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE} ${BENCHMARK_INTERFACE})

if(HAPI_KERNEL_LAUNCH_ENABLE_SAMPLING)
  target_compile_definitions(${BENCHMARK_NAME} PRIVATE BENCHMARK_ENABLE_SAMPLING)
endif()
