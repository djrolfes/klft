#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GDiracOperator.hpp"
#include "GLOBAL.hpp"
#include "Solver.hpp"

namespace klft {
template <typename DSpinorFieldType, typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT, typename,
                    typename> class _Solver,
          template <typename, typename> class DiracOpT>
struct pionCorrelator_per_Index {
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert(rank == DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank &&
                    Nc == DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc,
                "Rank and Nc must match between gauge, adjoint, and fermion "
                "field types.");
  using FermionField = typename DSpinorFieldType::type;
  using GaugeField = typename DGaugeFieldType::type;
  using DiracOperator =
      DiracOperator<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  using Solver = _Solver<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  const GaugeField gauge_fild;
  const FermionField source;
  FermionField drain;
};

template <typename DSpinorFieldType, typename DGaugeFieldType,
          template <template <typename, typename> class DiracOpT, typename,
                    typename> class _Solver,
          template <typename, typename> class DiracOpT>
// wrong return type
real_t PionCorrelator(
    const typename DGaugeFieldType::type& g_in,
    const diracParams<DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank,
                      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim>&
        params,
    const real_t& tol) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  using DiracOperator =
      DiracOperator<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  using Solver = _Solver<DiracOpT, DSpinorFieldType, DGaugeFieldType>;
  DiracOperator dirac_op(h.gauge_field, params);
  for (size_t i = 0; i < Nc * RepDim; i++) {
    deviceSpinorPointSource<Nc, RepDim> source(g_in.dimensions,
                                               IndexArray<rank>{0, 0, 0, 0}, i)
        FermionField x(dims, complex_t(0.0, 0.0));
    FermionField x0(dims, complex_t(0.0, 0.0));
    Solver solver(source, x, dirac_op);
    solver.template solve<Tags::TagDdaggerD>(x0, tol);
    auto prop = dirac_op.apply template <Tags::TagDdagger>
                (solver.x);

    // at the end vecotor with length Nt, maybe new view with only one dimension
    // to do the device Reduction
  }
}

}  // namespace klft
