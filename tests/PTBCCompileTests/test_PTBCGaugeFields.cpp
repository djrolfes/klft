// test_devicePTBCGaugeField.cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <iostream>

// Include the header(s) that define the devicePTBCGaugeField classes.
// For example, if you have one header that collects all these definitions:
// #include "devicePTBCGaugeField.hpp"
//
// Otherwise, include each header as needed:
#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/klft.hpp"
#include "../../include/PTBCGaugeField.hpp"


// We'll assume that our classes are in the klft namespace.
using namespace klft;

int main(int argc, char* argv[]) {
  // Initialize Kokkos.
  Kokkos::initialize(argc, argv);
  {
    std::cout << "\n=== Testing devicePTBCGaugeField (4D) ===\n";
    // Dimensions for 4D field:
    index_t L0 = 8, L1 = 8, L2 = 8, L3 = 8;
    // Set an initial complex value (e.g., identity type if that makes sense, here use (1,0))
    complex_t init_val(1.0, 0.0);
    index_t defect_length = 2;
    real_t cr = 0.5; // some real number for the defect

    // Instantiate the 4D gauge field with Nc = 3 (for example)
    devicePTBCGaugeField<4, 3> gauge4D(L0, L1, L2, L3, init_val, defect_length, cr);

    // Launch a parallel_for to print one field element for mu = 0
    Kokkos::fence();

    std::cout << "\n=== Testing devicePTBCGaugeField3D (3D) ===\n";
    // Dimensions for 3D field:
    index_t L0_3D = 8, L1_3D = 8, L2_3D = 8;
    devicePTBCGaugeField3D<3, 3> gauge3D(L0_3D, L1_3D, L2_3D, init_val, defect_length, cr);
    Kokkos::fence();

    std::cout << "\n=== Testing devicePTBCGaugeField2D (2D) ===\n";
    // Dimensions for 2D field:
    index_t L0_2D = 8, L1_2D = 8;
    devicePTBCGaugeField2D<2, 3> gauge2D(L0_2D, L1_2D, init_val, defect_length, cr);
    Kokkos::fence();
  }
  Kokkos::finalize();
  return 0;
}
