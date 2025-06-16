#pragma once
#include "AdjointField.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "Kokkos_Macros.hpp"
#include "Tuner.hpp"
#include <type_traits>

namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType>
struct HamiltonianField {
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;

  GaugeField gauge_field;
  AdjointField adjoint_field;

  HamiltonianField() = default;

  HamiltonianField(const GaugeField &_gauge_field,
                   const AdjointField &_adjoint_field)
      : gauge_field(_gauge_field), adjoint_field(_adjoint_field) {}

  //   real_t kinetic_energy() {
  //     real_t kinetic_energy = 0.0;
  // tune_and_launch_for<rank>("kinetic_energy", BulkPolicy,
  // KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t,
  // const int mu) {
  //       Adjoint tmp = adjoint_field.get_adjoint(x,y,z,t,mu);
  //       local_sum += 0.5*tmp.norm2();
  //     }, kinetic_energy);
  //     return kinetic_energy;
  //   }

  // TODO:write the randomize_momentum function using  the randSUNAdj call
  template <class RNG> void randomize_momentum(RNG rng) {
    tune_and_launch_for<rank>(
        "randomize_momentum", IndexArray<rank>{0}, adjoint_field.dimension,
        KOKKOS_LAMBDA(const int x, const int y, const int z, const int t,
                      const int mu) {
          auto generator = rng.get_state();
          Adjoint U;
          U.get_random(generator);
          adjoint_field.set_adjoint(x, y, z, t, mu, U);
          rng.free_state(generator);
        });
  }
};

} // namespace klft
