#include <assert.h>

#include <iomanip>

#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "Reducer.hpp"
#include "SUN.hpp"
#include "Spinor.hpp"
#include "Tuner.hpp"
using namespace klft;

#define HLINE "=========================================================\n"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int N = 12671;
    Kokkos::View<int[2][N]> a("label");
    Reducer::ArrayType<int, 2> array;
    tune_and_launch_for<2>(
        "init_label",
        IndexArray<2>{
            0,
            0,
        },
        IndexArray<2>{2, N}, KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          a(i0, i1) = i0 * N + i1 + 1;
        });
    Kokkos::parallel_reduce(
        "Reductor", Policy<2>(IndexArray<2>{}, IndexArray<2>{2, N}),
        KOKKOS_LAMBDA(const int& i0, const int& i1,
                      Reducer::ArrayType<int, 2>& upd) {
          int ndx = i0 % 2;  // sum all of the i%4 entries (divide total by 4)
          upd.array[ndx] += a(i0, i1);
        },
        Kokkos::Sum<Reducer::ArrayType<int, 2>>(array));
    printf("Result: %ld, %ld \n ", array.array[0], array.array[1]);
    printf("Expected Results : %d, %d \n", int((N * N + N) / 2),
           int((4 * N * N + 2 * N) / 2) - int((N * N + N) / 2));
  }
  Kokkos::finalize();
  return 0;
}