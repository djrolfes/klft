# MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# --------------------------
# Set and check QUDA_HOME
# --------------------------
if("${QUDA_HOME}" STREQUAL "")
  message(FATAL_ERROR "QUDA_HOME must be defined")
endif()

include_directories(SYSTEM ${QUDA_HOME}/include)
link_directories(${QUDA_HOME}/lib)

# Find QUDA library
find_library(QUDA_LIB NAMES quda PATHS ${QUDA_HOME}/lib)
if(NOT QUDA_LIB)
  message(FATAL_ERROR "Could not find QUDA library in ${QUDA_HOME}/lib")
endif()

# --------------------------
# Build Executables
# --------------------------
add_executable(test_wflow test_wflow.cpp)
target_link_libraries(test_wflow PUBLIC Kokkos::kokkos)

add_executable(quda_wflow_test quda_wflow_test.cpp)
target_link_libraries(quda_wflow_test 
  PUBLIC 
    Kokkos::kokkos 
    ${QUDA_LIB} 
    ${MPI_CXX_LIBRARIES}
)
