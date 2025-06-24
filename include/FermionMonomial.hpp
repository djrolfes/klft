#pragma once
#include "GDiracOperator.hpp"
#include "GLOBAL.hpp"
#include "Monomial.hpp"
#include "Solver.hpp"
#include "SpinorFieldLinAlg.hpp"

#define SQRT2INV \
  0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506  // Oeis A010503
namespace klft {
template <typename DiracOperator,
          class Solver,
          class RNG,
          typename DFermionField,
          typename DGaugeFieldType,
          typename DAdjFieldType>
class FermionMonomial : public Monomial<DGaugeFieldType, DAdjFieldType> {
  static_assert(isDeviceFermionFieldType<DFermionField>::value);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DFermionField>::RepDim;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank ==
                 DeviceFermionFieldTypeTraits<DFermionField>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc) ==
                    DeviceFermionFieldTypeTraits<DFermionField>::Nc);

 public:
  DFermionField phi;
  const diracParams<rank, RepDim> params;
  const real_t tol;
  RNG rng;
  FermionMonomial(DFermionField& _phi,
                  const diracParams<rank, RepDim>& params_,
                  const real_t& tol_,
                  RNG& RNG_,
                  unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale),
        phi(_phi),
        params(params_),
        RNG(RNG_),
        tol(tol_) {
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_FERMION;
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    auto dims = h.gauge_field.dimensions;

    DFermionField R(dims, rng, 0, SQRT2INV);
    DiracOperator dirac_op(h.gauge_field, params);
    this->phi = dirac_op.applyD(R);

    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        spinor_norm_sq<rank, Nc, RepDim>(R);
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    auto dims = h.gauge_field.dimensions;

    DFermionField x(dims, complex_t(0.0, 0.0));
    DFermionField x0(dims, complex_t(0.0, 0.0));
    DiracOperator dirac_op(h.gauge_field, params);
    Solver solver(this->phi, x, dirac_op);
    const DFermionField chi = solver.solve(x0, this->tol);

    Monomial<DFermionField, DAdjFieldType>::H_new =
        spinor_dot_product<rank, Nc, RepDim>(this->phi, chi).real();
  }
};
}  // namespace klft
