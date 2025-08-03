#include <assert.h>

#include <iomanip>

#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "SUN.hpp"
#include "Spinor.hpp"
using namespace klft;

#define HLINE "=========================================================\n"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
  auto rng = random_pool.get_state();
  SUN<2> a;
  randSUN(a, rng, 0.1);
  random_pool.free_state(rng);
  print_SUN(a);
  Kokkos::finalize();
  return 0;
}