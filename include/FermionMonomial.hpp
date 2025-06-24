#pragma once
#include "GDiracOperator.hpp"
#include "GLOBAL.hpp"
#include "Monomial.hpp"
#include "Solver.hpp"
#include "SpinorFieldLinAlg.hpp"

#define SQRT2INV \
  0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506  // Oeis A010503
namespace klft {
template <typename DiracOperator, class Solver, class RNGType,
          typename DFermionFieldType, typename DGaugeFieldType,
          typename DAdjFieldType>
class FermionMonomial : public Monomial<DGaugeFieldType, DAdjFieldType> {
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
  using FermionField = typename DFermionFieldType::type;

 public:
  FermionField phi;
  const diracParams<rank, RepDim> params;
  const real_t tol;
  RNGType rng;
  FermionMonomial(FermionField _phi, const diracParams<rank, RepDim>& params_,
                  const real_t& tol_, RNGType& RNG_, unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale),
        phi(_phi),
        params(params_),
        rng(RNG_),
        tol(tol_) {
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_FERMION;
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    auto dims = h.gauge_field.dimensions;

    FermionField R(dims, rng, 0, SQRT2INV);
    DiracOperator dirac_op(h.gauge_field, params);
    this->phi = dirac_op.applyD(R);

    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        spinor_norm_sq<rank, Nc, RepDim>(R);
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    auto dims = h.gauge_field.dimensions;

    FermionField x(dims, complex_t(0.0, 0.0));
    FermionField x0(dims, complex_t(0.0, 0.0));
    DiracOperator dirac_op(h.gauge_field, params);
    Solver solver(this->phi, x, dirac_op);
    solver.solve(x0, this->tol);
    const FermionField chi = solver.x;

    Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
        spinor_dot_product<rank, Nc, RepDim>(this->phi, chi).real();
  }
};
}  // namespace klft
