#include "DiracOperator.hpp"
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "PropagatorMatrix.hpp"
#include "Solver.hpp"
#include "Spinor.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "SpinorPointSource.hpp"
#include "Tuner.hpp"
namespace klft {

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT,
                    typename,
                    typename> class _Solver,
          template <typename, typename> class DiracOpT>
std::vector<real_t> PionCorrelator(const typename DGaugeFieldType::type& g_in,
                                   const diracParams& params,
                                   const real_t& tol) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  using SpinorFieldSource =
      typename WithSpinorFieldKind<DSpinorFieldType,
                                   SpinorFieldKind::PointSource>::type;
  using SpinorField = typename DSpinorFieldType::type;
  using DiracOperator = DiracOpT<DSpinorFieldType, DGaugeFieldType>;
  using Solver = _Solver<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  printf("Kappa: %f\n", params.kappa);
  DiracOperator dirac_op(g_in, params);
  auto Nt = g_in.field.extent(3);

  std::vector<real_t> result_vec(Nt);

  if constexpr (rank == 4) {
    size_t Vs = g_in.dimensions[0] * g_in.dimensions[1] * g_in.dimensions[2];
    SpinorField x(g_in.dimensions, 0);
    SpinorField x0(g_in.dimensions, 0);
    for (index_t alpha0 = 0; alpha0 < Nc * RepDim; alpha0++) {
      SpinorFieldSource source(g_in.dimensions, IndexArray<rank>{}, alpha0);
      Solver solver(source, x, dirac_op);
      solver.template solve<Tags::TagDdaggerD>(x0, tol);
      auto prop = dirac_op.template apply<Tags::TagDdagger>(solver.x);
      for (size_t i3 = 0; i3 < g_in.dimensions[3]; i3++) {
        real_t res = 0.0;
        Kokkos::parallel_reduce(
            "Reductor",
            Policy<rank - 1>(
                IndexArray<rank - 1>{},
                IndexArray<rank - 1>{g_in.dimensions[0], g_in.dimensions[1],
                                     g_in.dimensions[2]}),
            KOKKOS_LAMBDA(const size_t& i0, const size_t& i1, const size_t& i2,
                          real_t& upd) { upd += sqnorm(prop(i0, i1, i2, i3)); },
            res);
        Kokkos::fence();
        result_vec[i3] += (res / Vs);
      }
    }
  }
  return result_vec;
}

}  // namespace klft
