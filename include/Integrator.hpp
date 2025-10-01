//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/
#pragma once
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"

namespace klft {

typedef enum IntegratorType_s { LEAPFROG = 0, LP_LEAPFROG } IntegratorType;

// template <class UpdatePosition, class UpdateMomentum>
class Integrator : public std::enable_shared_from_this<Integrator> {
 public:
  Integrator() = delete;
  virtual ~Integrator() = default;

  Integrator(const size_t n_steps_,
             const bool outermost_,
             std::shared_ptr<Integrator> nested_,
             std::shared_ptr<UpdatePosition> update_q_,
             std::shared_ptr<UpdateMomentum> update_p_)
      : n_steps(n_steps_),
        outermost(outermost_),
        nested(nested_),
        update_q(update_q_),
        update_p(update_p_) {};

  virtual void integrate(const real_t tau, const bool last_step) const = 0;
  virtual void halfstep(const real_t tau) const = 0;

  std::shared_ptr<Integrator> nested;
  std::shared_ptr<UpdatePosition> update_q;
  std::shared_ptr<UpdateMomentum> update_p;
  const size_t n_steps;
  const bool outermost;
};

// template <class UpdatePosition, class UpdateMomentum>
class LeapFrog : public Integrator {  // <UpdatePosition, UpdateMomentum> {
 public:
  LeapFrog() = delete;

  LeapFrog(const size_t n_steps_,
           const bool outermost_,
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
    if (outermost) {
      halfstep(tau);
    }
    const real_t eps = tau / n_steps;
    for (size_t i = 0; i < n_steps - 1; ++i) {
      if (nested) {
        Kokkos::Profiling::pushRegion("Nested Integrator Hot Loop");
        nested->integrate(eps, false);
        Kokkos::Profiling::popRegion();
      } else {
        update_q->update(eps);
      }
      update_p->update(eps);
    }
    if (nested) {
      Kokkos::Profiling::pushRegion("Nested Integrator");
      if (outermost) {
        nested->integrate(eps, true);
      } else {
        nested->integrate(eps, last_step);
      }
      Kokkos::Profiling::popRegion();
    } else {
      update_q->update(eps);
    }
    if (!last_step && !outermost)
      update_p->update(eps);
    if (outermost)
      halfstep(tau);
  }

};  // class LeapFrog

}  // namespace klft
