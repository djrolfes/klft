#pragma once
#include "FermionParams.hpp"
#include "GDiracOperator.hpp"
#include "IndexHelper.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "UpdateMomentum.hpp"
namespace klft {

template <typename DFermionField,
          typename DGaugeFieldType,
          typename DAdjFieldType,
          class Derived>
class UpdateMomentumFermion : public UpdateMomentum {
  static_assert(isDeviceFermionFieldType<DFermionField>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DFermionField>::RepDim;
  static_assert(rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank &&
                    rank == DeviceFermionFieldTypeTraits<DFermionField>::Rank &&
                    Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc &&
                    Nc == DeviceFermionFieldTypeTraits<DFermionField>::Nc,
                "Rank and Nc must match between gauge, adjoint, and fermion "
                "field types.");

 public:
  // Define Tags for different Operators
  struct HWilsonDiracOperatorTag {};

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  GaugeFieldType gauge_field;
  AdjFieldType momentum;
  const diracParams<rank, RepDim> params;
  // Assumes Spinor field has been inverted i.e \chi = (DD\dagger)^-1 \eta,
  // where \eta = D R, where R gaussian random field.
  SpinorFieldType& chi;
  SpinorFieldType chi_alt;

  real_t eps;

  UpdateMomentumFermion() = delete;
  ~UpdateMomentumFermion() = default;

  UpdateMomentumFermion(SpinorFieldType& chi_,
                        GaugeFieldType& gauge_field_,
                        AdjFieldType& adjoint_field_,
                        const diracParams<rank, RepDim>& params_)
      : UpdateMomentum(0),
        chi(chi_),
        gauge_field(gauge_field_),
        momentum(adjoint_field_),
        params(params_),
        eps(0.0) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(HWilsonDiracOperatorTag,
                                              const Indices... Idcs) const {
    // Update the momentum of the fermion field
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);
      auto first_term =
          (-xp.second * complex_t(0, 1 / 2)) *
          ((this->params.gamma_id - this->params.gammas[mu]) *
           (this->gauge_field(Idcs..., mu) * this->chi_alt(xp.first)));
      auto second_term =
          (xm.second * complex_t(0, 1 / 2)) *
          ((this->params.gamma_id + this->params.gammas[mu]) *
           (conj(this->gauge_field(Idcs..., mu)) * this->chi_alt(Idcs...)));
      auto total_term =
          conj(chi(Idcs...)) * first_term + conj(chi(xm.first)) * second_term;

      momentum(Idcs..., mu) -=
          eps * -2 *
          traceT(
              realSUN(total_term));  // in leap frog therese the minus sign here
    }
  }

  void update(const real_t step_size) override {
    eps = step_size;
    IndexArray<rank> start;
    Derived D(gauge_field, this->params);
    chi_alt = D.applyDdagger(chi);

    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }

    // launch the kernel
    tune_and_launch_for<rank, HWilsonDiracOperatorTag>(
        "UpdateMomentumFermion", start, params.dimensions, *this);
    Kokkos::fence();
  }
};

}  // namespace klft
