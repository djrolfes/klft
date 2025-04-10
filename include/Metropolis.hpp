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

// this file defines metropolis sweeps for different fields and actions

#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "SUN.hpp"
#include "Tuner.hpp"
#include "Metropolis_Params.hpp"
#include "Gauge_Util.hpp"

namespace klft
{
  // here we define functions for the metropolis update
  // the function sweep metropolis must be overloaded
  // for different fields and actions
  // by default, the function must return the number of 
  // accepted updates

  // metropolis sweep for the 4D gauge field with SUN gauge
  // the parameter oddeven is used to indicate whether
  // the sweep is for odd or even sites
  template <size_t Nd, size_t Nc, class RNG>
  size_t sweep_Metropolis_eo(deviceGaugeField<Nd,Nc> g_in,
                          const MetropolisParams &params, RNG &rng,
                          const bool oddeven) {
    // get the necessary metropolis parameters
    constexpr size_t rank = 4;
    const index_t nHits = params.nHits;
    const real_t beta = params.beta;
    const real_t delta = params.delta;
    const IndexArray<rank> start = {0, 0, 0, 0};

    // to sweep over odd/even sites,
    // we divide the lattice into two sublattices
    // by dividing the t (last) dimension into odd and even
    const IndexArray<rank> end = {params.L0, params.L1, params.L2, (index_t)(params.L3 / 2)};

    // define number of accepted updates
    // store it in a 4D field for each site
    // and reduce it at the end
    ScalarField nAccepted(Kokkos::view_alloc(Kokkos::WithoutInitializing, "nAccepted"), params.L0, params.L1, params.L2, params.L3);
    // initialize the number of accepted updates
    Kokkos::deep_copy(nAccepted, 0.0);

    // define the functor id based on odd or even sweep
    std::string functor_id = "sweep_Metropolis_GaugeField";
    if (oddeven) {
      functor_id += "_odd";
    } else {
      functor_id += "_even";
    }

    // tune and launch the kernel
    // since the first call to the kernel will tune it,
    // the first nAccepted will be garbage
    // so the user should do a warmup run before
    // calling this function
    tune_and_launch_for<rank>(functor_id, start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        index_t nAcc_per_site = 0;
        // get the rng state
        auto generator = rng.get_state();
        SUN<Nc> r;
        // get the last index based on odd/even
        const index_t i3_oe = oddeven ? 2*i3 + 1 : 2*i3;
        // iterate over mu
        #pragma unroll
        for(index_t mu = 0; mu < Nd; ++mu) {
          // get the staple
          const SUN<Nc> staple = g_in.staple(i0,i1,i2,i3_oe,mu);
          // do number of hits
          for(index_t hit = 0; hit < nHits; ++hit) {
            // generate a random SUN matrix
            randSUN(r, generator, delta);
            // get old link
            const SUN<Nc> U_old = g_in(i0,i1,i2,i3_oe,mu);
            // calculate the new link
            const SUN<Nc> U_new = U_old * r;
            // calculate delta S
            const real_t dS = -(beta/static_cast<real_t>(Nc))
                             * (trace(U_new * staple).real()
                              - trace(U_old * staple).real());
            // accept or reject the update
            bool accept = dS < 0.0;
            if (!accept) {
              accept = (generator.drand(0.0, 1.0) < Kokkos::exp(-dS));
            }
            if (accept) {
              // update the link
              g_in(i0,i1,i2,i3_oe,mu) = restoreSUN(U_new);
              // increment the number of accepted updates
              nAcc_per_site++;
            }
          }
        }
        // free the rng state
        rng.free_state(generator);
        // store the number of accepted updates
        nAccepted(i0,i1,i2,i3_oe) = static_cast<real_t>(nAcc_per_site);
      });
    Kokkos::fence();
    real_t nAcc_total = 0;
    // reduce the number of accepted updates
    Kokkos::parallel_reduce("reduce_nAccepted", Policy<rank>(start, end), 
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3, real_t &nAcc) {
        const index_t i3_oe = oddeven ? 2*i3 + 1 : 2*i3;
        // sum over all sites
        nAcc += static_cast<size_t>(nAccepted(i0,i1,i2,i3_oe));
      }, Kokkos::Sum<real_t>(nAcc_total));
    return static_cast<size_t>(nAcc_total);
  }

  // create a wrapper to perform the sweep for the gauge field
  // for both odd and even sites
  template <size_t Nd, size_t Nc, class RNG>
  size_t sweep_Metropolis(deviceGaugeField<Nd,Nc> g_in,
                          const MetropolisParams &params, RNG &rng) {
    // perform the sweep for odd and even sites
    const size_t nAcc_odd = sweep_Metropolis_eo<Nd,Nc,RNG>(g_in, params, rng, false);
    const size_t nAcc_even = sweep_Metropolis_eo<Nd,Nc,RNG>(g_in, params, rng, true);
    return nAcc_odd + nAcc_even;
  }


}