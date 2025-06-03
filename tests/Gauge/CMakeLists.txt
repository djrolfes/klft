# MPI

option(COMPILE_QUDA_TESTS "Compile tests that require QUDA" OFF)

# --------------------------
# Set and check QUDA_HOME if needed
# --------------------------
if(COMPILE_QUDA_TESTS)
  if("${QUDA_HOME}" STREQUAL "")
    message(FATAL_ERROR "QUDA_HOME must be defined when COMPILE_QUDA_TESTS is ON")
  endif()

	find_package(MPI REQUIRED)
	include_directories(SYSTEM ${MPI_INCLUDE_PATH})

  include_directories(SYSTEM ${QUDA_HOME}/include)
  link_directories(${QUDA_HOME}/lib)

  # Find QUDA library
  find_library(QUDA_LIB NAMES quda PATHS ${QUDA_HOME}/lib)
  if(NOT QUDA_LIB)
    message(FATAL_ERROR "Could not find QUDA library in ${QUDA_HOME}/lib")
  endif()
endif()

# --------------------------
# Build Executables
# --------------------------
add_executable(test_wflow test_wflow.cpp)
target_link_libraries(test_wflow PUBLIC Kokkos::kokkos)


if(COMPILE_QUDA_TESTS)
  add_executable(quda_wflow_test quda_wflow_test.cpp)
  target_link_libraries(quda_wflow_test 
    PUBLIC 
      Kokkos::kokkos 
      ${QUDA_LIB} 
      ${MPI_CXX_LIBRARIES}
  )
endif()
