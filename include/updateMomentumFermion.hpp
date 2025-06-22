#pragma once
#include "FermionParams.hpp"
#include "GDiracOperator.hpp"
#include "IndexHelper.hpp"
#include "UpdateMomentum.hpp"
namespace klft {

template <size_t rank, size_t Nc, size_t RepDim, class Derived>
class UpdateMomentumFermion : public UpdateMomentum {
 public:
  // Define Tags for different Operators
  struct HWilsonDiracOperator {};

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  SpinorFieldType spinor_field;
  GaugeFieldType gauge_field;
  AdjFieldType momentum;
  const diracParams<rank, Nc, RepDim> params;
  // Assumes Spinor field has been inverted i.e \chi = (DD\dagger)^-1 \eta,
  // where \eta = D R, where R gaussian random field.
  SpinorFieldType& chi;
  SpinorFieldType& chi_alt;
  const IndexArray<rank> dimensions;
  real_t eps;

  UpdateMomentumFermion() = delete;
  ~UpdateMomentumFermion() = default;

  UpdateMomentumFermion(SpinorFieldType& chi_, SpinorFieldType& chi_alt_,
                        GaugeFieldType& gauge_field_,
                        AdjFieldType& adjoint_field_,
                        const IndexArray<rank>& dimensions_)
      : chi(chi_),
        gauge_field(gauge_field_),
        momentum(adjoint_field_),
        dimensions(dimensions_),
        eps(0.0) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(HWilsonDiracOperator,
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
      auto first_term = (-xp.second * complex_t(0, 1 / 2)) *
                        (this->params.gamma_id - this->params.gammas[mu]) *
                        this->gauge_field(Idcs..., mu) * this->chi_alt(xp);
      auto second_term = (xm.second * complex_t(0, 1 / 2)) *
                         (this->params.gamma_id + this->params.gammas[mu]) *
                         conj(this->gauge_field(Idcs..., mu)) *
                         this->chi_alt(Idcs...);
      auto total_term =
          conj(chi(Idcs...)) * first_term + conj(chi(xm.first)) * second_term;

      momentum(Idcs...) -=
          eps *
          traceT(-2 *
                 realSpinor(
                     total_term));  // in leap frog therese the minus sign here
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
    tune_and_launch_for<rank, Derived::type>("UpdateMomentumFermion", start,
                                             dimensions, *this);
    Kokkos::fence();
  }
};

}  // namespace klft
