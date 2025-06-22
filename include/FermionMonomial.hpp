#pragma once
#include "GDiracOperator.hpp"
#include "GLOBAL.hpp"
#include "Monomial.hpp"
#include "Solver.hpp"
#include "SpinorFieldLinAlg.hpp"

#define SQRT2INV \
  0.707106781186547524400844362104849039284835937688474036588339868995366239231053519425193767163820786367506  // Oeis A010503
namespace klft {
template <typename DiracOperator, class Solver, class RNG, size_t rank,
          size_t Nc, size_t RepDim, typename DGaugeFieldType,
          typename DAdjFieldType>
class FermionMonomial
    : public Monomial<DeviceSpinorFieldType<rank, Nc, RepDim>, DAdjFieldType> {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank ==
                 DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc ==
                 DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;

 public:
  SpinorFieldType phi;
  const diracParams<rank, Nc, RepDim> params;
  const real_t tol;
  RNG rng;
  FermionMonomial(SpinorFieldType& _phi,
                  const diracParams<rank, Nc, RepDim>& params_,
                  const real_t& tol_, RNG& RNG_, unsigned int _time_scale)
      : Monomial<DeviceSpinorFieldType<rank, Nc, RepDim>, DAdjFieldType>(
            _time_scale),
        phi(_phi),
        params(params_),
        RNG(RNG_),
        tol(tol_) {
    Monomial<DeviceSpinorFieldType<rank, Nc, RepDim>,
             DAdjFieldType>::monomial_type = KLFT_MONOMIAL_FERMION;
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    auto dims = h.gauge_field.dimensions;

    SpinorFieldType R(dims, rng, 0, SQRT2INV);
    DiracOperator dirac_op(h.gauge_field, params);
    this->phi = dirac_op.applyD(R);

    Monomial<DeviceSpinorFieldType<rank, Nc, RepDim>, DAdjFieldType>::H_old =
        spinor_norm_sq<rank, Nc, RepDim>(R);
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    auto dims = h.gauge_field.dimensions;

    SpinorFieldType x(dims, complex_t(0.0, 0.0));
    SpinorFieldType x0(dims, complex_t(0.0, 0.0));
    DiracOperator dirac_op(h.gauge_field, params);
    Solver solver(this->phi, x, dirac_op);
    const SpinorFieldType chi = solver.solve(x0, this->tol);

    Monomial<DeviceSpinorFieldType<rank, Nc, RepDim>, DAdjFieldType>::H_new =
        spinor_dot_product<rank, Nc, RepDim>(this->phi, chi).real();
  }
};
}  // namespace klft
