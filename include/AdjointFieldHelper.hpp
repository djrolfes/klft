#pragma once
#include "AdjointField.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
namespace klft {
template <typename DAdjFieldType, class RNG>
void randomize_field(typename DAdjFieldType::type &field, RNG &rng) {
  size_t constexpr Nd = DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank;
  size_t constexpr Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;
  if constexpr (Nd == 4) {
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
  }
  if constexpr (Nd == 3) {
    tune_and_launch_for(
        "randomize_adj_field", IndexArray<Nd>{0}, field.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(field(i0, i1, i2, mu), generator);
          }
          rng.free_state(generator);
        });
  }
  if constexpr (Nd == 2) {
    tune_and_launch_for(
        "randomize_adj_field", IndexArray<Nd>{0}, field.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(field(i0, i1, mu), generator);
          }
          rng.free_state(generator);
        });
  }

  Kokkos::fence();
}

template <typename DAdjFieldType>
void flip_sign(typename DAdjFieldType::type &field) {
  size_t constexpr Nd = DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank;
  size_t constexpr Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;
  if constexpr (Nd == 4) {
    tune_and_launch_for(
        "randomize_adj_field", IndexArray<Nd>{0}, field.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            flip_sign<Nc>(field(i0, i1, i2, i3, mu));
          }
        });
  }
  if constexpr (Nd == 3) {
    tune_and_launch_for(
        "randomize_adj_field", IndexArray<Nd>{0}, field.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            flip_sign<Nc>(field(i0, i1, i2, mu));
          }
        });
  }
  if constexpr (Nd == 2) {
    tune_and_launch_for(
        "randomize_adj_field", IndexArray<Nd>{0}, field.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            flip_sign<Nc>(field(i0, i1, mu));
          }
        });
  }

  Kokkos::fence();
}
} // namespace klft
