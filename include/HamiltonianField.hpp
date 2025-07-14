#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_Macros.hpp"
#include "decl/Kokkos_Declare_OPENMP.hpp"
#include "impl/Kokkos_Profiling.hpp"

namespace klft {

template <typename DAdjFieldType> struct KineticEnergyFunctor {
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;
  constexpr static size_t rank = DeviceAdjFieldTypeTraits<DAdjFieldType>::rank;
  using AdjointField = typename DAdjFieldType::type;
  const AdjointField adjoint_field;

  KineticEnergyFunctor(const AdjointField &field) : adjoint_field(field) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(index_t i0, index_t i1, index_t i2, index_t i3,
                  real_t &rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 4; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, i2, i3, mu));
    }
    rtn += tmp;
  }

  // Rank-3 version
  KOKKOS_INLINE_FUNCTION void operator()(index_t i0, index_t i1, index_t i2,
                                         real_t &rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 3; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, i2, mu));
    }
    rtn += tmp;
  }

  // Rank-2 version
  KOKKOS_INLINE_FUNCTION void operator()(index_t i0, index_t i1,
                                         real_t &rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 2; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, mu));
    }
    rtn += tmp;
  }
};

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

  HamiltonianField() = delete;

  HamiltonianField(std::unique_ptr<GaugeField> g,
                   std::unique_ptr<AdjointField> a)
      : _gauge_field(std::move(g)), _adjoint_field(std::move(a)) {}

  GaugeField &gauge_field() { return *_gauge_field.get(); }
  const GaugeField &gauge_field() const { return *_gauge_field.get(); }
  AdjointField &adjoint_field() { return *_adjoint_field.get(); }
  const AdjointField &adjoint_field() const { return *_adjoint_field.get(); }

  real_t kinetic_energy() {
    real_t kinetic_energy = 0.0;
    constexpr size_t r = rank; // for clarity
    const auto &dim = adjoint_field().dimensions;
    auto functor = KineticEnergyFunctor<DAdjFieldType>(adjoint_field());

    if constexpr (r == 4) {
      Kokkos::MDRangePolicy<Kokkos::Rank<4>> rp(
          {0, 0, 0, 0}, {dim[0], dim[1], dim[2], dim[3]});
      Kokkos::parallel_reduce("kinetic_energy", rp, functor, kinetic_energy);
    } else if constexpr (r == 3) {
      Kokkos::MDRangePolicy<Kokkos::Rank<3>> rp({0, 0, 0},
                                                {dim[0], dim[1], dim[2]});
      Kokkos::parallel_reduce("kinetic_energy", rp, functor, kinetic_energy);
    } else if constexpr (r == 2) {
      Kokkos::MDRangePolicy<Kokkos::Rank<2>> rp({0, 0}, {dim[0], dim[1]});
      Kokkos::parallel_reduce("kinetic_energy", rp, functor, kinetic_energy);
    } else {
      static_assert(r <= 4, "Unsupported rank in kinetic_energy");
    }

    return kinetic_energy;
  }

  template <class RNG> void randomize_momentum(RNG &rng) {
    adjoint_field().template randomize_field<RNG>(rng);
  }

private:
  std::unique_ptr<GaugeField> _gauge_field;
  std::unique_ptr<AdjointField> _adjoint_field;
};

} // namespace klft
