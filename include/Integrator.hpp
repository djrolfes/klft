#pragma once
#include "HMC_Params.hpp"
#include "Solver.hpp"
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"
#include "updateMomentumFermion.hpp"

namespace klft {

typedef enum IntegratorType_s { LEAPFROG = 0, LP_LEAPFROG } IntegratorType;

// template <class UpdatePosition, class UpdateMomentum>
class Integrator : public std::enable_shared_from_this<Integrator> {
public:
  Integrator() = delete;
  virtual ~Integrator() = default;

  Integrator(const size_t n_steps_, const bool outermost_,
             std::shared_ptr<Integrator> nested_,
             std::shared_ptr<UpdatePosition> update_q_,
             std::shared_ptr<UpdateMomentum> update_p_)
      : n_steps(n_steps_), outermost(outermost_), nested(nested_),
        update_q(update_q_), update_p(update_p_) {};

  virtual void integrate(const real_t tau, const bool last_step) const = 0;
  virtual void halfstep(const real_t tau) const = 0;

  std::shared_ptr<Integrator> nested;
  std::shared_ptr<UpdatePosition> update_q;
  std::shared_ptr<UpdateMomentum> update_p;
  const size_t n_steps;
  const bool outermost;
};

// template <class UpdatePosition, class UpdateMomentum>
class LeapFrog : public Integrator { // <UpdatePosition, UpdateMomentum> {
public:
  LeapFrog() = delete;

  LeapFrog(const size_t n_steps_, const bool outermost_,
           std::shared_ptr<Integrator> nested_,
           std::shared_ptr<UpdatePosition> update_q_,
           std::shared_ptr<UpdateMomentum> update_p_)
      : Integrator(n_steps_, outermost_, nested_, update_q_, update_p_) {};

  ~LeapFrog() override = default;

  void halfstep(const real_t tau) const override {
    const real_t eps = tau / n_steps;
    update_p->update(eps * 0.5);
    if (nested)
      nested->halfstep(eps);
  }

  void integrate(const real_t tau, const bool last_step) const override {
    if (outermost)
      halfstep(tau);
    const real_t eps = tau / n_steps;
    for (size_t i = 0; i < n_steps - 1; ++i) {
      if (nested) {
        nested->integrate(eps, false);
      } else {
        update_q->update(eps);
      }
      update_p->update(eps);
    }
    if (nested) {
      if (outermost) {
        nested->integrate(eps, true);
      } else {
        nested->integrate(eps, last_step);
      }
    } else {
      update_q->update(eps);
    }
    if (!last_step && !outermost)
      update_p->update(eps);
    if (outermost)
      halfstep(tau);
  }

}; // class LeapFrog
//

template <typename DGaugeFieldType, typename DAdjFieldType,
          typename DSpinorFieldType>
std::shared_ptr<Integrator> createIntegrator(
    typename DGaugeFieldType::type &g_in, typename DAdjFieldType::type &a_in,
    typename DSpinorFieldType::type &s_in,
    const Integrator_Params &integratorParams,
    const GaugeMonomial_Params &gaugeMonomialParams,
    const FermionMonomial_Params &fermionParams, const int &resParsef) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  constexpr const size_t Nd = rank;
  constexpr const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  // Create the integrator based on the type
  if (integratorParams.monomials.empty()) {
    printf("Error: Integrator must have at least one monomial\n");
    return nullptr;
  }
  // startingpoit of integrator chain
  std::shared_ptr<Integrator> nested_integrator = nullptr;
  for (const auto &monomial : integratorParams.monomials) {
    std::shared_ptr<Integrator> integrator = nullptr;
    if (monomial.level == 0) {
      // if the level is 0, we create a new integrator with nullptr as inner
      // integrator
      if (gaugeMonomialParams.level == 0) {
        UpdatePositionGauge<Nd, Nc, k> update_q(g_in, a_in);
        UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType> update_p(
            g_in, a_in, gaugeMonomialParams.beta);
        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr,
              std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p));
        } else {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr,
              std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p));
        }

      } else if (fermionParams.level == 0 && resParsef > 0) {
        // if the level is 0, we create a new integrator with nullptr as inner
        // integrator
        if (fermionParams.RepDim == 4) {
          auto diracParams =
              getDiracParams<rank>(g_in.dimensions, fermionParams);
          UpdatePositionGauge<Nd, Nc, k> update_q(g_in, a_in);
          UpdateMomentumFermion<DSpinorFieldType, DGaugeFieldType,
                                DAdjFieldType, CGSolver, HWilsonDiracOperator>
              update_p(s_in, g_in, a_in, diracParams, fermionParams.tol);

          if (monomial.type == "Leapfrog") {
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
                std::make_shared<UpdateMomentumFermion<
                    DSpinorFieldType, DGaugeFieldType, DAdjFieldType,

                    CGSolver, HWilsonDiracOperator>>(update_p));

          } else {
            printf(
                "Warning: Integrator %s isn't available, falllback Leapfrog ",
                monomial.type.c_str());
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
                std::make_shared<UpdateMomentumFermion<
                    DSpinorFieldType, DGaugeFieldType, DAdjFieldType, CGSolver,
                    HWilsonDiracOperator>>(update_p));
          }
        } else {
          printf("Error: Fermion RepDim must be 4\n");
          return nullptr;
        }
      }
      nested_integrator = integrator;
    } else if (gaugeMonomialParams.level == monomial.level) {
      // if the level is the same, we create a new integrator with the
      // previous one as inner integrator
      UpdatePositionGauge<Nd, Nc, k> update_q(g_in, a_in);
      UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType> update_p(
          g_in, a_in, gaugeMonomialParams.beta);
      if (monomial.type == "Leapfrog") {
        integrator = std::make_shared<LeapFrog>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p));

      } else {
        integrator = std::make_shared<LeapFrog>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p));
      }

    } else if (fermionParams.level == monomial.level && resParsef > 0) {
      // if the level is 0, we create a new integrator with nullptr as inner
      // integrator
      if (fermionParams.RepDim == 4) {
        auto diracParams = getDiracParams<rank>(g_in.dimensions, fermionParams);

        UpdatePositionGauge<Nd, Nc, k> update_q(g_in, a_in);
        UpdateMomentumFermion<DSpinorFieldType, DGaugeFieldType, DAdjFieldType,
                              CGSolver, HWilsonDiracOperator>
            update_p(s_in, g_in, a_in, diracParams, fermionParams.tol);

        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
              std::make_shared<UpdateMomentumFermion<
                  DSpinorFieldType, DGaugeFieldType, DAdjFieldType, CGSolver,
                  HWilsonDiracOperator>>(update_p));
        } else {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc, k>>(update_q),
              std::make_shared<UpdateMomentumFermion<
                  DSpinorFieldType, DGaugeFieldType, DAdjFieldType, CGSolver,
                  HWilsonDiracOperator>>(update_p));
        }
      } else {
        printf("Error: Fermion RepDim must be 4\n");
        return nullptr;
      }
    }
    nested_integrator = integrator;
  }

  return nested_integrator;
}

} // namespace klft
