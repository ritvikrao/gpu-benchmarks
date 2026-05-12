function(add_charm_executable TARGET_NAME SRC_FILE CI_FILE)
  find_program(_CHARMC_EXECUTABLE NAMES charmc HINTS "${CHARM_ROOT}/bin" REQUIRED)

  get_filename_component(_ci_basename ${CI_FILE} NAME_WE)

  # derive CamelCase basename from underscore_separated name (e.g. hapi_kernel_launch -> HapiKernelLaunch)
  string(REGEX MATCHALL "[^_]+" _tokens ${_ci_basename})
  set(_ci_camel "")
  foreach(_tok IN LISTS _tokens)
    string(SUBSTRING ${_tok} 0 1 _first)
    string(TOUPPER ${_first} _first_up)
    string(SUBSTRING ${_tok} 1 -1 _rest)
    set(_ci_camel "${_ci_camel}${_first_up}${_rest}")
  endforeach()

  set(_camel_decl "${CMAKE_CURRENT_BINARY_DIR}/${_ci_camel}.decl.h")
  set(_camel_def  "${CMAKE_CURRENT_BINARY_DIR}/${_ci_camel}.def.h")
  set(_out_decl   "${CMAKE_CURRENT_BINARY_DIR}/${_ci_basename}.decl.h")
  set(_out_def    "${CMAKE_CURRENT_BINARY_DIR}/${_ci_basename}.def.h")

  # Run charmc which produces CamelCase generated headers.
  add_custom_command(
    OUTPUT ${_camel_decl} ${_camel_def}
    COMMAND ${_CHARMC_EXECUTABLE} -language charm++ ${CI_FILE}
    DEPENDS ${CI_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  # Copy/rename the CamelCase outputs to the lowercase filenames expected by sources
  add_custom_command(
    OUTPUT ${_out_decl} ${_out_def}
    COMMAND ${CMAKE_COMMAND} -E copy ${_camel_decl} ${_out_decl}
    COMMAND ${CMAKE_COMMAND} -E copy ${_camel_def} ${_out_def}
    DEPENDS ${_camel_decl} ${_camel_def}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
  )

  # Determine Charm include dir from charmc location.
  get_filename_component(_charm_bin_dir ${_CHARMC_EXECUTABLE} DIRECTORY)
  set(_charm_include_dir "${_charm_bin_dir}/../include")

  set(_benchmark_compiler "${CMAKE_CXX_COMPILER}")
  if(BENCHMARK_BACKEND STREQUAL "CUDA")
    set(_kokkos_wrapper "${KOKKOS_ROOT}/bin/nvcc_wrapper")
    if(EXISTS "${_kokkos_wrapper}")
      set(_benchmark_compiler "${_kokkos_wrapper}")
    endif()
  endif()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}
    COMMAND ${_benchmark_compiler}
            -std=c++20
            $<$<STREQUAL:${BENCHMARK_BACKEND},CUDA>:-extended-lambda>
            $<$<STREQUAL:${BENCHMARK_BACKEND},CUDA>:-Wext-lambda-captures-this>
            $<$<STREQUAL:${BENCHMARK_BACKEND},CUDA>:-DBENCHMARK_USE_CUDA>
            $<$<STREQUAL:${BENCHMARK_BACKEND},HIP>:-DBENCHMARK_USE_HIP>
            -I${KOKKOS_INCLUDE_DIR}
            -I${CMAKE_CURRENT_BINARY_DIR}
            -I${_charm_include_dir}
            -c ${SRC_FILE}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.o
    COMMAND ${_CHARMC_EXECUTABLE}
            ${BACKEND_CHARM_FLAG}
            -language charm++
            $<$<STREQUAL:${BENCHMARK_BACKEND},CUDA>:-lcudart>
            $<$<STREQUAL:${BENCHMARK_BACKEND},CUDA>:-lcuda>
            -L${KOKKOS_LIBRARY_DIR}
            -lkokkoscore
            -lkokkoscontainers
            -lkokkosalgorithms
            -lkokkossimd
            -o ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}
            ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.o
    DEPENDS ${SRC_FILE} ${_out_decl} ${_out_def}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
    COMMAND_EXPAND_LISTS
  )

  add_custom_target(${TARGET_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endfunction()
