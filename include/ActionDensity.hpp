#pragma once
#include "FieldStrengthTensor.hpp"

namespace klft {

template <typename DGaugeFieldType,
          typename FSTTag =
              typename FieldStrengthTensor<DGaugeFieldType>::CloverDef>
struct ActionDensityFunctor {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t Nd = rank;
  // constexpr static GaugeFieldKind kind =
  // DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

  using GaugeField = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;

  GaugeField gauge_field;
  FieldType density_per_site;
  const FieldStrengthTensor<DGaugeFieldType> fst;

  ActionDensityFunctor(const GaugeField _gauge_field)
      : gauge_field(_gauge_field),
        density_per_site(_gauge_field.dimensions, real_t(0.0)),
        fst(_gauge_field) {}

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void
  operator()(const indexType i0, const indexType i1, const indexType i2,
             const indexType i3) const {
    Kokkos::Array<Kokkos::Array<RealMatrix, Nd>, Nd> C;

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        // get the clover C_munu
        RealMatrix C_munu = fst(FSTTag{}, i0, i1, i2, i3, mu, nu);
        C[mu][nu] = C_munu;
        C[nu][mu] = -C_munu;
      }
    }

#pragma unroll
    for (int mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (int nu = mu + 1; nu < Nd; ++nu) {
        density_per_site(i0, i1, i2, i3) += trace(C[mu][nu] * C[mu][nu]);
      }
    }
  }
};

template <typename DGaugeFieldType>
real_t getActionDensity(const typename DGaugeFieldType::type g_in) {
  // calculate the density E according to (3.1) in
  // https://arxiv.org/pdf/1006.4518
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>(),
                "action density requires a device gauge field type.");
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  // constexpr static const size_t Nc =
  //     DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  // constexpr static const GaugeFieldKind kind =
  //     DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

  using FieldType = typename DeviceFieldType<Nd>::type;
  ActionDensityFunctor<DGaugeFieldType> ADFunctor(g_in);
  // define the functor
  tune_and_launch_for<Nd>("Calculate densityEAsym", IndexArray<Nd>{0},
                          g_in.dimensions, ADFunctor);
  Kokkos::fence();

  real_t density = 2 * Kokkos::real(ADFunctor.density_per_site.avg());
  Kokkos::fence();

  return density;
}
} // namespace klft
