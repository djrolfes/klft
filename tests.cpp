#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "GaugeGroup.hpp"      // Defines klft::SU3<T> and its methods.
#include "AdjointGroup.hpp"    // Defines klft::AdjointSU3<T> and SU3Generators.

using namespace klft;

template <typename T>
void test_su3_adjoint_roundtrip() {
  // Use the default execution space's device type:
  using device_type = typename Kokkos::DefaultExecutionSpace::device_type;
  using RNGPool = Kokkos::Random_XorShift64_Pool<device_type>;

  unsigned seed = 11293802u;
  RNGPool rng_pool(seed);

  // Get an RNG state from the pool.
  auto rng = rng_pool.get_state();

  AdjointSU3<T> randAdjU;
  randAdjU.get_random(rng);
  randAdjU = 0.02*randAdjU;

  // Create a SU3 element and perturb it randomly.
  SU3<T> U;
  U.get_random(rng, T(0.5));  // Fill U with a random perturbation of amplitude 0.5.
  Kokkos::printf("randU det: (%f, %f)\n", U.det().real(), U.det().imag());

  // Release the RNG state.
  rng_pool.free_state(rng);

  for (int i = 0; i<8; i++){
    std::cout << "Small AdjU element " << i <<": " << randAdjU.v[i] << std::endl;
  }

  //for (int i=0; i<9; i++){
  //  std::cout << "SU3 Matrix Element "<< i<<": "<< U.v[i] << std::endl;
  //}
  // Optionally: apply a small random perturbation to U
  // (You can write your own random perturbation routine, e.g., U += delta * random_SU3_lie_element)
  // For simplicity, we'll just work with the identity here.
  
  // Print original invariants.
  //auto trU = U.trace();
  //auto detU = U.det();
  //std::cout << "Original SU3:" << std::endl;
  //std::cout << "  Trace: " << trU << std::endl;
  //std::cout << "  Determinant: " << detU << std::endl;

  // Construct the adjoint representation from U.
  //AdjointSU3<T> adjU(U);
  //auto params = adjU.v;
  //for (int i=0; i<8; i++){
  //std::cout << "Adjoint: " << params[i] << std::endl;
  //}
  // Convert the adjoint back to an SU3 matrix.
  // asMatrix() returns a Kokkos::Array<Kokkos::complex<T>, 9>; use that to construct a new SU3.
  auto U_rec_array = exp(randAdjU);
  Kokkos::printf("randU_rec det: (%f, %f)\n", U_rec_array.det().real(), U_rec_array.det().imag());

  for (int i=0; i<9; i++){
    std::cout << "U_rec_array Matrix element " << i << ": " << U_rec_array.v[i] << std::endl;

  }


  SU3<T> U_rec(U_rec_array);  // assuming SU3 has a constructor that takes a 9-element array

  // Compute invariants of the reconstructed SU3 matrix.
  auto trU_rec = U_rec.trace();
  auto detU_rec = U_rec.det();
  std::cout << "Reconstructed SU3:" << std::endl;
  std::cout << "  Trace: " << trU_rec << std::endl;
  std::cout << "  Determinant: " << detU_rec << std::endl;


  AdjointSU3<T> AdjointAgain(U_rec);
  for (int i = 0; i< 8; i++){
    std::cout << "Back in AdjointSU3 ("<< i<<"): " << AdjointAgain.v[i] << std::endl;
  }

  // Compare the original and reconstructed SU3 element.
  // For a perfect round-trip, they should be equal.
  // In practice, you may see small differences due to numerical errors.
  //T diff_norm = T(0);
//for (int i = 0; i < 9; i++){
//  auto diff = U.v[i] - U_rec.v[i];
//  diff_norm += diff.real()*diff.real() + diff.imag()*diff.imag();
//}
//diff_norm = std::sqrt(diff_norm);
//std::cout << "Norm of difference between original and reconstructed SU3: " << diff_norm << std::endl;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Testing SU3 <-> AdjointSU3 roundtrip transformation..." << std::endl;
    test_su3_adjoint_roundtrip<double>();
  }
  Kokkos::finalize();
  return 0;
}
