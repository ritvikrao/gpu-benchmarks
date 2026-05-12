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

add_charm_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE} ${BENCHMARK_INTERFACE})
