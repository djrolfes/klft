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

// define structs for initializing gauge fields

#pragma once
#include "GLOBAL.hpp"
#include "Tuner.hpp"
#include "SUN.hpp"

namespace klft
{

  template <size_t Nd, size_t Nc>
  struct deviceGaugeField {

    deviceGaugeField() = delete;

    // initialize all sites to a given value
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     const complex_t init) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, init);
    }

    // initialize all links to a given SUN matrix
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     const SUN<Nc> &init) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, init);
    }

    // initialize all links to a random SUN matrix
    template <class RNG>
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng, const real_t delta) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, rng, delta);
    }

    // initialize all sites to a random value
    template <class RNG>
    deviceGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng) : dimensions({L0, L1, L2, L3}) {
      do_init(L0, L1, L2, L3, field, rng);
    }

    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<4>("init_deviceGaugeField", 
        IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0,i1,i2,i3,mu)[c1][c2] = init;
              }
            }
          }
        });
      Kokkos::fence();
    }

    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, const SUN<Nc> &init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<4>("init_deviceGaugeField",
        IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0,i1,i2,i3,mu) = init;
          }
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, RNG &rng, const real_t delta) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<4>("init_deviceGaugeField",
        IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0,i1,i2,i3,mu), generator, delta);
          }
          rng.free_state(generator);
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, RNG &rng) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<4>("init_deviceGaugeField",
        IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0,i1,i2,i3,mu)[c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                                   generator.drand(-1.0, 1.0));
              }
            }
          }
        });
      Kokkos::fence();
    }

    GaugeField<Nd,Nc> field;
    const IndexArray<4> dimensions;

    // define accessors for the field
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const indexType i0, const indexType i1, const indexType i2, const indexType i3, const index_t mu) const {
      return field(i0,i1,i2,i3,mu);
    }
  
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const indexType i0, const indexType i1, const indexType i2, const indexType i3, const index_t mu) {
      return field(i0,i1,i2,i3,mu);
    }

    // define accessors with 4D Kokkos array
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const Kokkos::Array<indexType,4> site, const index_t mu) const {
      return field(site[0], site[1], site[2], site[3], mu);
    }
  
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const Kokkos::Array<indexType,4> site, const index_t mu) {
      return field(site[0], site[1], site[2], site[3], mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(const Kokkos::Array<indexType,4> site, const index_t mu) const {
      // this only works if Nd == 4
      assert(Nd == 4);
      // get the indices
      const index_t i0 = site[0];
      const index_t i1 = site[1];
      const index_t i2 = site[2];
      const index_t i3 = site[3];
      // temporary SUN matrix to store the staple
      SUN<Nc> temp = zeroSUN<Nc>();
      // get the x + mu indices
      const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const index_t i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
      const index_t i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
      // positive directions
      #pragma unroll
      for(index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        // do nothing for mu = nu
        if (nu == mu) continue;
        // get the x + nu indices
        const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
        const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
        const index_t i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;
        const index_t i3pnu = nu == 3 ? (i3 + 1) % dimensions[3] : i3;
        // get the staple
        temp += field(i0pmu,i1pmu,i2pmu,i3pmu,nu) * conj(field(i0pnu,i1pnu,i2pnu,i3pnu,mu))
              * conj(field(i0,i1,i2,i3,nu));
      } // loop over nu
      // negative directions
      #pragma unroll
      for(index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        // do nothing for mu = nu
        if (nu == mu) continue;
        // get the x + mu - nu indices
        const index_t i0pmu_mnu = nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
        const index_t i1pmu_mnu = nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
        const index_t i2pmu_mnu = nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;
        const index_t i3pmu_mnu = nu == 3 ? (i3pmu - 1 + dimensions[3]) % dimensions[3] : i3pmu;
        // get the x - nu indices
        const index_t i0mnu = nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
        const index_t i1mnu = nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
        const index_t i2mnu = nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;
        const index_t i3mnu = nu == 3 ? (i3 - 1 + dimensions[3]) % dimensions[3] : i3;
        // get the staple
        temp += conj(field(i0pmu_mnu,i1pmu_mnu,i2pmu_mnu,i3pmu_mnu,nu)) 
              * conj(field(i0mnu,i1mnu,i2mnu,i3mnu,mu)) * field(i0mnu,i1mnu,i2mnu,i3mnu,nu);
      } // loop over nu
      return temp;
    }
    
  };

  template <size_t Nd, size_t Nc>
  struct deviceGaugeField3D {

    deviceGaugeField3D() = delete;

    // initialize all sites to a given value
    deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2, 
                       const complex_t init) : dimensions({L0, L1, L2}) {
      do_init(L0, L1, L2, field, init);
    }

    // initialize all links to a given SUN matrix
    deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2, 
                       const SUN<Nc> &init) : dimensions({L0, L1, L2}) {
      do_init(L0, L1, L2, field, init);
    }

    // initialize all links to a random SUN matrix
    template <class RNG>
    deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2, 
                       RNG &rng, const real_t delta) : dimensions({L0, L1, L2}) {
      do_init(L0, L1, L2, field, rng, delta);
    }

    // initialize all sites to a random value
    template <class RNG>
    deviceGaugeField3D(const index_t L0, const index_t L1, const index_t L2, 
                       RNG &rng) : dimensions({L0, L1, L2}) {
      do_init(L0, L1, L2, field, rng);
    }

    void do_init(const index_t L0, const index_t L1, const index_t L2, 
                 GaugeField3D<Nd, Nc> &V, complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
      tune_and_launch_for<3>("init_deviceGaugeField3D",
        IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, i2, mu)[c1][c2] = init;
              }
            }
          }
        });
      Kokkos::fence();
    }

    void do_init(const index_t L0, const index_t L1, const index_t L2, 
                 GaugeField3D<Nd, Nc> &V, const SUN<Nc> &init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
      tune_and_launch_for<3>("init_deviceGaugeField3D",
        IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = init;
          }
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, const index_t L2, 
                 GaugeField3D<Nd, Nc> &V, RNG &rng, const real_t delta) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
      tune_and_launch_for<3>("init_deviceGaugeField3D",
        IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0, i1, i2, mu), generator, delta);
          }
          rng.free_state(generator);
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, const index_t L2, 
                 GaugeField3D<Nd, Nc> &V, RNG &rng) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
      tune_and_launch_for<3>("init_deviceGaugeField3D",
        IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, i2, mu)[c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                                     generator.drand(-1.0, 1.0));
              }
            }
          }
        });
      Kokkos::fence();
    }

    GaugeField3D<Nd, Nc> field;
    const IndexArray<3> dimensions;

    // define accessors for the field
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const indexType i0, const indexType i1, const indexType i2, const index_t mu) const {
      return field(i0, i1, i2, mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const indexType i0, const indexType i1, const indexType i2, const index_t mu) {
      return field(i0, i1, i2, mu);
    }

    // define accessors with 3D Kokkos array
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const Kokkos::Array<indexType,3> site, const index_t mu) const {
      return field(site[0], site[1], site[2], mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const Kokkos::Array<indexType,3> site, const index_t mu) {
      return field(site[0], site[1], site[2], mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(const Kokkos::Array<indexType,3> site, const index_t mu) const {
      // this only works if Nd == 3
      assert(Nd == 3);
      // get the indices
      const index_t i0 = site[0];
      const index_t i1 = site[1];
      const index_t i2 = site[2];
      // temporary SUN matrix to store the staple
      SUN<Nc> temp = zeroSUN<Nc>();
      // get the x + mu indices
      const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const index_t i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
  
      // positive directions
      #pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        if (nu == mu) continue; // skip if mu == nu
        const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
        const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
        const index_t i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;

        temp += field(i0pmu, i1pmu, i2pmu, nu) * conj(field(i0pnu, i1pnu, i2pnu, mu))
              * conj(field(i0, i1, i2, nu));
      }
  
      // negative directions
      #pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        if (nu == mu) continue; // skip if mu == nu
        const index_t i0pmu_mnu = nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
        const index_t i1pmu_mnu = nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
        const index_t i2pmu_mnu = nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;

        const index_t i0mnu = nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
        const index_t i1mnu = nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
        const index_t i2mnu = nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;

        temp += conj(field(i0pmu_mnu, i1pmu_mnu, i2pmu_mnu, nu)) 
              * conj(field(i0mnu, i1mnu, i2mnu, mu)) * field(i0mnu, i1mnu, i2mnu, nu);
      }
  
      return temp;
    }

<<<<<<< HEAD
  };

  template <size_t Nd, size_t Nc>
  struct deviceGaugeField2D {

    deviceGaugeField2D() = delete;

    // initialize all sites to a given value
    deviceGaugeField2D(const index_t L0, const index_t L1, 
                       const complex_t init) : dimensions({L0, L1}) {
      do_init(L0, L1, field, init);
=======
    KOKKOS_INLINE_FUNCTION Group get_symanzik_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group staple(0.0);
      Group U1, U2, U3, U4, U5;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_pm_mu = {x,y,z,t};
      Kokkos::Array<int,4> site_pm_two_mu = {x,y,z,t};
      site_pm_mu[this->array_dims[mu]] = (site_pm_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      site_pm_two_mu[this->array_dims[mu]] = (site_pm_two_mu[this->array_dims[mu]] + 2) % this->dims[mu];
      Kokkos::Array<int,4> site_pm_nu_mu = {x,y,z,t};
      site_pm_nu_mu[this->array_dims[mu]] = (site_pm_nu_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_pm_nu_mu[this->array_dims[nu]] = (site_pm_nu_mu[this->array_dims[nu]] + 1) % this->dims[nu];
        U1 = get_link(site_pm_mu,mu);
        U2 = get_link(site_pm_two_mu,nu);
        U3 = get_link(site_pm_nu_mu,mu);
        U4 = get_link(site_pm_nu, mu);
        U5 = get_link(site, nu);
        staple += U1*U2*dagger(U3)*dagger(U4)*dagger(U5);
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu_mu[this->array_dims[nu]] = (site_pm_nu_mu[this->array_dims[nu]] - 1+this->dims[nu]) % this->dims[nu];
      }

      //site_pm_mu[this->array_dims[mu]] = (site_pm_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      site_pm_two_mu[this->array_dims[mu]] = (site_pm_two_mu[this->array_dims[mu]] - 3 + this->dims[mu]) % this->dims[mu]; //mu=-1 maybe use another name here
      site_pm_nu_mu[this->array_dims[mu]] = (site_pm_nu_mu[this->array_dims[mu]] - 2 + this->dims[mu]) % this->dims[mu]; //mu = -1

      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_pm_nu_mu[this->array_dims[nu]] = (site_pm_nu_mu[this->array_dims[nu]] + 1 ) % this->dims[nu];
        U1 = get_link(site_pm_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site_pm_nu_mu,mu);
        U4 = get_link(site_pm_two_mu, nu);
        U5 = get_link(site_pm_two_mu, mu);
        staple += U1*dagger(U2)*dagger(U3)*dagger(U4)*U5;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu_mu[this->array_dims[nu]] = (site_pm_nu_mu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }

      site_pm_mu[this->array_dims[mu]] = (site_pm_mu[this->array_dims[mu]] -2 + this->dims[mu]) % this->dims[mu];//mu=-1
      site_pm_two_mu[this->array_dims[mu]] = (site_pm_two_mu[this->array_dims[mu]] - 1 + this->dims[mu]) % this->dims[mu]; //mu=-2

      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        //site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_pm_nu_mu[this->array_dims[nu]] = (site_pm_nu_mu[this->array_dims[nu]] + 1 ) % this->dims[nu];
        U1 = get_link(site_pm_nu_mu,nu);
        U2 = get_link(site_pm_mu,nu);
        U3 = get_link(site_pm_two_mu,mu);
        U4 = get_link(site_pm_two_mu, nu);
        U5 = get_link(site_pm_mu, nu);
        staple += dagger(U1)*dagger(U2)*dagger(U3)*U4*U5;
        //site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu_mu[this->array_dims[nu]] = (site_pm_nu_mu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }

      return staple;
    }

    void copy(const GaugeField<T,Group,Ndim,Nc> &in) {
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          Kokkos::deep_copy(this->gauge[mu][i], in.gauge[mu][i]);
        }
      }
>>>>>>> 00ef82da6dfbc1fa2c29da97c6c151d7f27eeefd
    }

    // initialize all links to a given SUN matrix
    deviceGaugeField2D(const index_t L0, const index_t L1, 
                       const SUN<Nc> &init) : dimensions({L0, L1}) {
      do_init(L0, L1, field, init);
    }

    // initialize all links to a random SUN matrix
    template <class RNG>
    deviceGaugeField2D(const index_t L0, const index_t L1, 
                       RNG &rng, const real_t delta) : dimensions({L0, L1}) {
      do_init(L0, L1, field, rng, delta);
    }

    // initialize all sites to a random value
    template <class RNG>
    deviceGaugeField2D(const index_t L0, const index_t L1, 
                       RNG &rng) : dimensions({L0, L1}) {
      do_init(L0, L1, field, rng);
    }

    void do_init(const index_t L0, const index_t L1, 
                 GaugeField2D<Nd, Nc> &V, complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
      tune_and_launch_for<2>("init_deviceGaugeField2D",
        IndexArray<2>{0, 0}, IndexArray<2>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, mu)[c1][c2] = init;
              }
            }
          }
        });
      Kokkos::fence();
    }

    void do_init(const index_t L0, const index_t L1, 
                 GaugeField2D<Nd, Nc> &V, const SUN<Nc> &init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
      tune_and_launch_for<2>("init_deviceGaugeField2D",
        IndexArray<2>{0, 0}, IndexArray<2>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = init;
          }
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 GaugeField2D<Nd, Nc> &V, RNG &rng, const real_t delta) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
      tune_and_launch_for<2>("init_deviceGaugeField2D",
        IndexArray<2>{0, 0}, IndexArray<2>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0, i1, mu), generator, delta);
          }
          rng.free_state(generator);
        });
      Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 GaugeField2D<Nd, Nc> &V, RNG &rng) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
      tune_and_launch_for<2>("init_deviceGaugeField2D",
        IndexArray<2>{0, 0}, IndexArray<2>{L0, L1},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            #pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
              #pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, mu)[c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                                 generator.drand(-1.0, 1.0));
              }
            }
          }
        });
      Kokkos::fence();
    }

    GaugeField2D<Nd, Nc> field;
    const IndexArray<2> dimensions;

    // define accessors for the field
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const indexType i0, const indexType i1, const index_t mu) const {
      return field(i0, i1, mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const indexType i0, const indexType i1, const index_t mu) {
      return field(i0, i1, mu);
    }

    // define accessors with 2D Kokkos array
    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const Kokkos::Array<indexType,2> site, const index_t mu) const {
      return field(site[0], site[1], mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &operator()(const Kokkos::Array<indexType,2> site, const index_t mu) {
      return field(site[0], site[1], mu);
    }

    template <typename indexType>
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(const Kokkos::Array<indexType,2> site, const index_t mu) const {
      // this only works if Nd == 2
      assert(Nd == 2);
      // get the indices
      const index_t i0 = site[0];
      const index_t i1 = site[1];
      // temporary SUN matrix to store the staple
      SUN<Nc> temp = zeroSUN<Nc>();
      // get the x + mu indices
      const index_t i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const index_t i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
  
      // positive directions
      #pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        if (nu == mu) continue; // skip if mu == nu
        const index_t i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
        const index_t i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;

        temp += field(i0pmu, i1pmu, nu) * conj(field(i0pnu, i1pnu, mu))
              * conj(field(i0, i1, nu));
      }
  
      // negative directions
      #pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
        if (nu == mu) continue; // skip if mu == nu
        const index_t i0pmu_mnu = nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
        const index_t i1pmu_mnu = nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;

        const index_t i0mnu = nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
        const index_t i1mnu = nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;

        temp += conj(field(i0pmu_mnu, i1pmu_mnu, nu)) 
              * conj(field(i0mnu, i1mnu, mu)) * field(i0mnu, i1mnu, nu);
      }
  
      return temp;
    }

  };

}
