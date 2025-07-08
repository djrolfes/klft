#pragma once
#include "FermionParams.hpp"
#include "GDiracOperator.hpp"
#include "IndexHelper.hpp"
#include "SpinorFieldLinAlg.hpp"
#include "UpdateMomentum.hpp"
namespace klft {

template <typename DFermionFieldType,
          typename DGaugeFieldType,
          typename DAdjFieldType,
          class Derived,
          class Solver>
class UpdateMomentumFermion : public UpdateMomentum {
  static_assert(isDeviceFermionFieldType<DFermionFieldType>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DFermionFieldType>::RepDim;
  static_assert(rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank &&
                    rank ==
                        DeviceFermionFieldTypeTraits<DFermionFieldType>::Rank &&
                    Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc &&
                    Nc == DeviceFermionFieldTypeTraits<DFermionFieldType>::Nc,
                "Rank and Nc must match between gauge, adjoint, and fermion "
                "field types.");

 public:
  // Define Tags for different Operators
  struct HWilsonDiracOperatorTag {};

  using FermionField = typename DFermionFieldType::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  GaugeFieldType& gauge_field;
  AdjFieldType& momentum;
  const diracParams<rank, RepDim> params;
  // \phi = D R, where R gaussian random field.
  FermionField& phi;

  FermionField chi;
  FermionField chi_alt;
  const real_t tol;
  real_t eps;

  UpdateMomentumFermion() = delete;
  ~UpdateMomentumFermion() = default;

  UpdateMomentumFermion(FermionField& phi_,
                        GaugeFieldType& gauge_field_,
                        AdjFieldType& adjoint_field_,
                        const diracParams<rank, RepDim>& params_,
                        const real_t& tol_)
      : UpdateMomentum(0),
        phi(phi_),
        gauge_field(gauge_field_),
        momentum(adjoint_field_),
        params(params_),
        eps(0.0),
        tol(tol_) {}

  // Implemntation of the force correspondig to the Hermitian Wilson dirac
  // Operator
  //* S_F  = \phi^{\dagger} * M^{-1} * \phi
  //* dS_F = - \phi^{\dagger} * M^{-1} * dM * M^{-1} * \phi = - \chi^{\dagger} *
  // dM * \chi
  //* where \chi = M^{-1} * \phi
  //* and M = Q*Q^{\dagger}
  // Using that dS_f = - \chi ^{\dagger} (\pdv{D}{U_\mu(x)}D^\dagger + D
  // \pdv{D^\dagger}{U_\mu(x)})\chi which can be simplified to -\chi^\dagger
  // (\pdv{D}{U_\mu(x)}D^\dagger+ (\pdv{D}{U_\mu(x)}D^\dagger)^\dagger) =
  // -2Re(\chi^\dagger \pdv{D}{U_\mu(x)}D^\dagger \chi)

  // The Derivative of the Hermtian Wilson Dirac Operator should be given as
  // \pdv{D(x,y)}{U_\mu(z)} = - \kappa \gamma_5 \delta_{x,z} (1-\gamma_mu)T_i
  // U_mu(z)\delta_{x+\mu,y}
  // +\kappa \gamma_5 \delta_{x+\mu,y} (1+\gamma_\mu) t_i
  // U_\mu(z)^\dagger\delta_{z,y} <- Have to Check this
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
          (xp.second * -this->params.kappa) *
          ((this->params.gamma_id - this->params.gammas[mu]) *
           (this->gauge_field(Idcs..., mu) * this->chi_alt(xp.first)));
      auto second_term =
          (xp.second * this->params.kappa) *
          ((this->params.gamma_id + this->params.gammas[mu]) *
           (conj(this->gauge_field(Idcs..., mu)) * this->chi_alt(Idcs...)));
      // The first multiplication makes the force matrix
      auto total_term =
          conj(chi(Idcs...)) * (this->params.gamma5 * first_term) +
          conj(chi(xp.first)) * (this->params.gamma5 * second_term);

      auto derv = 2 * this->eps * total_term;
      // if (Kokkos::Array<size_t, rank>{Idcs...} ==
      //     Kokkos::Array<size_t, rank>{0, 0, 0, 0}) {
      //   print_SUN(traceLessAntiHermitian(derv), "SUN Force Matrix ");
      //   print_SUNAdj(traceT(traceLessAntiHermitian(derv)), "Adj Force
      //   MAtrix");
      // }

      // Taking the Real part is handled by the traceT
      momentum(Idcs..., mu) -= traceT(traceLessAntiHermitian(
          derv));  // in leap frog therese the minus sign here
    }
  }

  void update(const real_t step_size) override {
    eps = step_size;

    IndexArray<rank> start;
    Derived D(gauge_field, this->params);
    FermionField x(this->params.dimensions, complex_t(0.0, 0.0));
    FermionField x0(this->params.dimensions, complex_t(0.0, 0.0));
    // print_spinor_int(this->phi(0, 0, 0, 0),
    //                  "Spinor s_in(0,0,0,0) in update Momentum Fermion");
    Solver solver(this->phi, x, D);
    if (KLFT_VERBOSITY > 4) {
      printf("Solving insde updateMomentumFermion:");
    }

    solver.solve(x0, this->tol);
    this->chi = solver.x;
    this->chi_alt = D.applyDdagger(this->chi);
    // print_spinor_int(solver.x(0, 0, 0, 0),
    //                  "solver.x after solve (should be the same as chi)");
    // print_spinor_int(this->chi(0, 0, 0, 0), "chi after solve");
    // print_SUN(gauge_field(0, 0, 0, 0, 0), "In GaugeField in
    // Fermionmomentum");
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    // auto before = this->momentum(0, 0, 0, 0, 0);

    // launch the kernel
    tune_and_launch_for<rank, HWilsonDiracOperatorTag>(
        "UpdateMomentumFermion", start, gauge_field.dimensions, *this);
    Kokkos::fence();
    // print_SUNAdj(this->momentum(0, 0, 0, 0, 0), "SUNAdj after update
    // Fermion");
  }
};

}  // namespace klft
