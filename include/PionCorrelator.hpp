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

template <typename DSpinorFieldType, typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT, typename,
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

  std::vector<real_t> result_vec(Nt - 1);

  if constexpr (rank == 4) {
    typename DevicePropagator<rank, Nc, RepDim>::type result(
        g_in.dimensions, complex_t(0.0, 0.0));
    size_t Vs = g_in.dimensions[0] * g_in.dimensions[1] * g_in.dimensions[2];
    for (index_t alpha0 = 0; alpha0 < Nc * RepDim; alpha0++) {
      SpinorFieldSource source(g_in.dimensions, IndexArray<rank>{}, alpha0);
      SpinorField x(g_in.dimensions, 0);
      SpinorField x0(g_in.dimensions, 0);
      Solver solver(source, x, dirac_op);
      solver.template solve<Tags::TagDdaggerD>(x0, tol);
      auto prop = dirac_op.template apply<Tags::TagDdagger>(solver.x);
      for (size_t i3 = 1; i3 < g_in.dimensions[3]; i3++) {
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
        result_vec[i3] += (res);
      }
    }
  }
  return result_vec;
}

template <typename DSpinorFieldType, typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT, typename,
                    typename> class _Solver,
          template <typename, typename> class DiracOpT>
std::vector<real_t> PionCorrelatorEO(
    const typename DGaugeFieldType::type& g_in, const diracParams& params,
    IndexArray<DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank> f_dims,
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
  DiracOperator dirac_op(g_in, params);
  auto Nt = g_in.field.extent(3);

  std::vector<real_t> result_vec;

  if constexpr (rank == 4) {
    typename DevicePropagator<rank, Nc, RepDim>::type result(
        g_in.dimensions, complex_t(0.0, 0.0));
    size_t Vs = g_in.dimensions[0] * g_in.dimensions[1] * g_in.dimensions[2];
    SpinorField x(f_dims, 0);
    SpinorField x0(f_dims, 0);
    SpinorField prop_even(f_dims, 0);
    SpinorField prop_odd(f_dims, 0);
    for (index_t alpha0 = 0; alpha0 < Nc * RepDim; alpha0++) {
      SpinorFieldSource source(f_dims, IndexArray<rank>{},
                               alpha0);  // even source
      Solver solver(SpinorField(source), x, dirac_op);
      if constexpr (std::is_same_v<Solver, CGSolver<DiracOpT, DSpinorFieldType,
                                                    DGaugeFieldType>>) {
        printf("CG currently not supported\n");
      }
      if constexpr (std::is_same_v<Solver, BiCGStab<DiracOpT, DSpinorFieldType,
                                                    DGaugeFieldType>>) {
        // BicvCGStab gives D^-1 directly
        solver.template solve<Tags::TagSe>(x0, tol);
        prop_even = solver.x;
        solver.reconstruct_solution_0(prop_odd);
      }
      tune_and_launch_for<rank>(
          "even_Propagator_population", IndexArray<rank>{0, 0, 0, 0}, f_dims,
          KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                        const index_t i3) {
            auto idx_gbl =
                index_half_to_full(IndexArray<rank>{i0, i1, i2, i3}, 0);
            add_inplace(result(idx_gbl), prop_even(i0, i1, i2, i3), alpha0);
          });
      tune_and_launch_for<rank>(
          "odd_Propagator_population", IndexArray<rank>{0, 0, 0, 0}, f_dims,
          KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                        const index_t i3) {
            auto idx_gbl =
                index_half_to_full(IndexArray<rank>{i0, i1, i2, i3}, 1);
            add_inplace(result(idx_gbl), prop_odd(i0, i1, i2, i3), alpha0);
          });
      Kokkos::fence();
      // Function
    }
    // at the end vecotor with length Nt, maybe new view with only one dimension
    // to do the device Reduction
    for (size_t i3 = 0; i3 < g_in.dimensions[3]; i3++) {
      real_t res = 0;

      Kokkos::parallel_reduce(
          "Reductor",
          Policy<rank - 1>(
              IndexArray<rank - 1>{},
              IndexArray<rank - 1>{g_in.dimensions[0], g_in.dimensions[1],
                                   g_in.dimensions[2]}),
          KOKKOS_LAMBDA(const size_t& i0, const size_t& i1, const size_t& i2,
                        real_t& upd) {
#pragma unroll
            for (size_t alpha = 0; alpha < Nc * RepDim; alpha++) {
#pragma unroll
              for (size_t beta = 0; beta < Nc * RepDim; beta++) {
                upd += (result(i0, i1, i2, i3)[beta][alpha] *
                        conj(result(i0, i1, i2, i3)[beta][alpha]))
                           .real();
              }
            }
          },
          res);
      Kokkos::fence();
      result_vec.push_back(res / static_cast<real_t>(Vs));
    }
  }
  return result_vec;
}

}  // namespace klft
