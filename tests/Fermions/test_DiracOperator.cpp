#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "GLOBAL.hpp"
// #include "FieldTypeHelper.hpp"
#include "../../include/GDiracOperator.hpp"
#include "../../include/GammaMatrix.hpp"
#include "../../include/SpinorField.hpp"
#include "../../include/SpinorFieldLinAlg.hpp"
#include "../../include/klft.hpp"
#define HLINE "=========================================================\n"

using namespace klft;
template <size_t Nc, size_t Nd>
void print_spinor(const Spinor<Nc, Nd>& s, const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t c = 0; c < Nc; ++c) {
    printf("  Color %zu:\n", c);
    for (size_t d = 0; d < Nd; ++d) {
      double re = s[c][d].real();
      double im = s[c][d].imag();
      printf("    [%zu] = (% .6f, % .6f i)\n", d, re, im);
    }
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int RETURNVALUE = 0;
  {
    setVerbosity(5);
    printf("%i", KLFT_VERBOSITY);
    printf("\n=== Testing DiracOperator SU(3)  ===\n");
    printf("\n= Testing hermiticity =\n");
    index_t L0 = 32, L1 = 32, L2 = 32, L3 = 32;
    auto gammas = get_gammas<4>();
    GammaMat<4> gamma5 = get_gamma5();
    diracParams<4, 4> params(IndexArray<4>{L0, L1, L2, L3}, gammas, gamma5,
                             -0.5);
    printf("Lattice Dimension %ix%ix%ix%i \n", L0, L1, L2, L3);
    printf("Generate SpinorFields...\n");

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    deviceSpinorField<3, 4> u(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    deviceSpinorField<3, 4> v(L0, L1, L2, L3, random_pool, 0, 1.0 / 1.41);
    real_t norm = spinor_norm_sq<4, 3, 4>(u);
    norm *= spinor_norm_sq<4, 3, 4>(v);
    norm = Kokkos::sqrt(norm);

    printf("Generating Random Gauge Config\n");
    deviceGaugeField<4, 3> gauge(L0, L1, L2, L3, random_pool, 1);
    printf("Instantiate DiracOperator...\n");
    HWilsonDiracOperator<DeviceSpinorFieldType<4, 3, 4>,
                         DeviceGaugeFieldType<4, 3>>
        D(gauge, params);
    printf("Apply DiracOperator...\n");

    deviceSpinorField Mu = D.template apply<Tags::TagD>(u);
    deviceGaugeField<4, 3> gaugeTrafo(L0, L1, L2, L3, random_pool, 1);
    tune_and_launch_for<4>(
        "Gauge Trafo", IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (size_t mu = 0; mu < 4; mu++) {
            // gauge(i0, i1, i2, i3, mu) = gaugeTrafo(i0, i1, i2, i3, mu);
            auto temp = gauge(i0, i1, i2, i3, mu);
            gauge(i0, i1, i2, i3, mu) -= temp;
            gauge(i0, i1, i2, i3, mu) -= temp;
          }
          // Transform spinor u, and Mu
        });

    deviceSpinorField Mv1 = D.template apply<Tags::TagD>(u);
    // deviceSpinorField<3, 4> Mu = apply_D<4, 3, 4>(u, gauge, gammas, -0.5);
    // deviceSpinorField<3, 4> Mv = apply_D<4, 3, 4>(v, gauge, gammas, -0.5);
    print_spinor(Mv1(0, 0, 0, 0), "Mv1");
    print_spinor(Mu(0, 0, 0, 0), "Mu");
    printf("Calculate Scalarproducts...\n");
    // auto r1 = spinor_dot_product<4, 3, 4>(u, Mv1);
    // auto r2 = spinor_dot_product<4, 3, 4>(Mu, v);

    // auto r = r1 - r2;

    // real_t r3 = Kokkos::sqrt(r.real() * r.real() + r.imag() * r.imag());
    // r3 /= norm;

    // if (r3 < 1e-14) {
    //   printf("Passed hermiticity test with %.21f \n", r3);
    // } else {
    //   printf("Error: didn't pass hermiticity test with %.21f \n", r3);
    //   RETURNVALUE++;
    // }
  }
  Kokkos::finalize();
  return RETURNVALUE;
}