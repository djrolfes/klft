#pragma once
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"
#ifdef ENABLE_DEBUG
#include <iostream>
#define DEBUG_LOG(msg)                                                         \
  do {                                                                         \
    std::cout << msg << std::endl;                                             \
  } while (0)
#else
#define DEBUG_LOG(msg)                                                         \
  do {                                                                         \
  } while (0)
#endif

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

    DEBUG_LOG("[halfstep]  τ=" << tau << "  ⇒  eps=" << eps
                               << "  |  update_p (½‑step) with ε/2="
                               << eps * 0.5);

    update_p->update(eps * real_t(0.5));

    if (nested) {
      DEBUG_LOG("[halfstep]  call nested->halfstep(ε)…");
      nested->halfstep(eps);
    }
  }

  //------------------------------------------------------------

  void integrate(const real_t tau, const bool last_step) const override {
    DEBUG_LOG("\n[integrate]  τ=" << tau << "  last_step=" << std::boolalpha
                                  << last_step);

    // ----- leading half‑step on the outermost level -----
    if (outermost) {
      DEBUG_LOG("[integrate]  outermost → leading halfstep() call");
      halfstep(tau);
    }

    const real_t eps = tau / n_steps;
    DEBUG_LOG("[integrate]  eps=" << eps << "  n_steps=" << n_steps);

    // ----- main loop (n_steps‑1 full steps) -----
    for (size_t i = 0; i < n_steps - 1; ++i) {
      DEBUG_LOG("[integrate]  loop step " << i + 1 << "/" << n_steps);

      if (nested) {
        DEBUG_LOG("[integrate]    nested->integrate(ε, false)");
        nested->integrate(eps, false);
      } else {
        DEBUG_LOG("[integrate]    update_q->update(ε)");
        update_q->update(eps);
      }

      DEBUG_LOG("[integrate]    update_p->update(ε)");
      update_p->update(eps);
    }

    // ----- final position update -----
    if (nested) {
      DEBUG_LOG(
          "[integrate]  final nested integrate (last/outermost handling)");
      if (outermost) {
        nested->integrate(eps, true);
      } else {
        nested->integrate(eps, last_step);
      }
    } else {
      DEBUG_LOG("[integrate]  final update_q->update(ε)");
      update_q->update(eps);
    }

    // ----- trailing momentum update for inner/middle levels -----
    if (!last_step && !outermost) {
      DEBUG_LOG("[integrate]  trailing update_p->update(ε) on inner level");
      update_p->update(eps);
    }

    // ----- trailing half‑step on the outermost level -----
    if (outermost) {
      DEBUG_LOG("[integrate]  outermost → trailing halfstep() call");
      halfstep(tau);
    }
  }

}; // class LeapFrog

} // namespace klft
