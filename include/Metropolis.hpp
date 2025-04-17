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
#include "FieldTypeHelper.hpp"
#include "IndexHelper.hpp"
#include "Metropolis_Params.hpp"

namespace klft
{
  // here we define functions for the metropolis update
  // the function sweep metropolis must be overloaded
  // for different fields and actions
  // by default, the function must return the number of 
  // accepted updates

  template <size_t rank, size_t Nc, class RNG>
  struct MetropolisGaugeField {
    // we strictly work with Nd = rank
    constexpr static const size_t Nd = rank;
    // define the gauge field type
    using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
    GaugeFieldType g_in;
    // define the scalar field type
    using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
    ScalarFieldType nAccepted;
    // define the rng
    const RNG rng;
    // define the parameters
    const MetropolisParams params;
    // lattice dimensions
    const IndexArray<rank> dimensions;
    // sublattice definitions
    const Kokkos::Array<bool, rank> oddeven;
    // constructor
    MetropolisGaugeField(const GaugeFieldType &g_in,
                         const MetropolisParams &params,
                         const IndexArray<rank> &dimensions,
                         const ScalarFieldType &nAccepted,
                         const Kokkos::Array<bool, rank> &oddeven,
                         const RNG &rng)
      : g_in(g_in), params(params), oddeven(oddeven), rng(rng),
        dimensions(dimensions), nAccepted(nAccepted) {}
    
    template <typename... Indices>
    KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
      index_t nAcc_per_site = 0;
      // get the rng state
      auto generator = rng.get_state();
      SUN<Nc> r;
      // get the site based on odd/even
      const IndexArray<rank> site = index_odd_even<rank,size_t>(Kokkos::Array<size_t,rank>{Idcs...}, oddeven);
      // iterate over mu (sequentially)
      for (index_t mu = 0; mu < Nd; ++mu) {
        // get the staple
        const SUN<Nc> staple = g_in.staple(site, mu);
        // do number of hits
        for (index_t hit = 0; hit < params.nHits; ++hit) {
          // generate a random SUN matrix
          randSUN(r, generator, params.delta);
          // get old link
          const SUN<Nc> U_old = g_in(site, mu);
          // calculate the new link
          const SUN<Nc> U_new = U_old * r;
          // calculate delta S
          const real_t dS = -(params.beta/static_cast<real_t>(Nc))
                           * (trace(U_new * staple).real()
                            - trace(U_old * staple).real());
          // accept or reject the update
          bool accept = dS < 0.0;
          if (!accept) {
            accept = (generator.drand(0.0, 1.0) < Kokkos::exp(-dS));
          }
          if (accept) {
            // update the link
            g_in(site, mu) = restoreSUN(U_new);
            // increment the number of accepted updates
            nAcc_per_site++;
          }
        }
      }
      // store the number of accepted updates
      nAccepted(Idcs...) += static_cast<real_t>(nAcc_per_site);
      // free the rng state
      rng.free_state(generator);
    }
  };

  // metropolis sweep for SUN gauge fields
  // returns acceptance rate
  template <size_t rank, size_t Nc, class RNG>
  real_t sweep_Metropolis(typename DeviceGaugeFieldType<rank, Nc>::type &g_in,
                          const MetropolisParams &params, const RNG &rng) {
    // this works strictly for Nd = rank
    constexpr static const size_t Nd = rank;
    // get start and end indices
    const auto & dimensions = g_in.field.layout().dimension;
    IndexArray<rank> start;
    IndexArray<rank> end;
    for (index_t i = 0; i < Nd; ++i) {
      start[i] = 0;
      end[i] = (index_t)(dimensions[i]/2);
    }

    // define a scalar field to store the number of accepted updates
    using ScalarFieldType = typename DeviceScalarFieldType<rank>::type;
    ScalarFieldType nAccepted(end, 0.0);

    // for N dimensional lattices, we need to divide each dimension 
    // into odd and even sites and launch the kernel for each sublattice
    // this results in 2^N sublattices
    for (index_t i = 0; i < std::pow(2, rank); ++i) {
      // define metropolis functor for this sublattice
      MetropolisGaugeField<rank, Nc, RNG> metropolis(g_in, params, end,
                           nAccepted, oddeven_array<rank>(i), rng);
      // launch the kernel
      tune_and_launch_for<rank>("sweep_Metropolis_GaugeField", start, end,
        metropolis);
      Kokkos::fence();
    }
    // reduce the number of accepted updates
    real_t nAcc_total = nAccepted.sum();
    Kokkos::fence();
    // normalize the acceptance rate
    real_t norm = 1.0;
    for (index_t i = 0; i < rank; ++i) {
      norm *= static_cast<real_t>(dimensions[i]);
    }
    norm *= static_cast<real_t>(Nd);
    norm *= static_cast<real_t>(params.nHits);
    nAcc_total /= norm;
    // return the acceptance rate
    return nAcc_total;
  }

}