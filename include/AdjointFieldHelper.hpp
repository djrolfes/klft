#pragma once
#include "AdjointField.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
namespace klft {
template <typename DAdjFieldType, class RNG>
void randomize_field(typename DAdjFieldType::type& field, RNG& rng) {
  size_t constexpr Nd = DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank;
  size_t constexpr Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;
  tune_and_launch_for(
      "randomize_adj_field", IndexArray<Nd>{0}, field.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                    const index_t i3) {
        auto generator = rng.get_state();
        for (index_t mu = 0; mu < Nd; ++mu) {
          randSUNAdj<Nc>(field(i0, i1, i2, i3, mu), generator);
        }
        rng.free_state(generator);
      });
  // Kokkos::fence();
}
}  // namespace klft
