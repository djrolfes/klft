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

// define structs for initializing ptbc gauge fields

#pragma once
#include "GLOBAL.hpp"
#include "Tuner.hpp"
#include "SUN.hpp"

namespace klft
{

  template <size_t Nd, size_t Nc>
  struct devicePTBCGaugeField {

    devicePTBCGaugeField() = delete;
    
    // should defect_length and cr be encompassed in a defect struct?
    devicePTBCGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     const complex_t init, const index_t _defect_length, 
                     const real_t cr) {
      do_init(L0, L1, L2, L3, field, init);
      do_init_defect(L0, L1, L2, L3, defectField, _defect_length, cr);
    }

    template <class RNG>
    devicePTBCGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng, const real_t delta, 
                     const index_t _defect_length, const real_t cr) {
      do_init(L0, L1, L2, L3, field, rng, delta);
      do_init_defect(L0, L1, L2, L3, defectField, _defect_length, cr);
    }

    template <class RNG>
    devicePTBCGaugeField(const index_t L0, const index_t L1, 
                     const index_t L2, const index_t L3, 
                     RNG &rng, const index_t _defect_length, 
                     const real_t cr) {
      do_init(L0, L1, L2, L3, field, rng);
      do_init_defect(L0, L1, L2, L3, defectField, _defect_length, cr);
    }

    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, complex_t init) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
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

    // without any defect information, set defect to be non-existent.
    void do_init_defect(const index_t L0, const index_t L1, 
                        const index_t L2, const index_t L3, 
                        LinkScalarField<Nd,Nc> &V) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
      #pragma unroll
      for (index_t mu = 0; mu < Nd; ++mu) {
          V(i0,i1,i2,i3,mu) = 1.0; 
        }
      });
      Kokkos::fence();
    }

    // This fixes the defect location according to (2.3) in 2404.14151 
    void do_init_defect(const index_t L0, const index_t L1, 
          const index_t L2, const index_t L3, 
          LinkScalarField<Nd,Nc> &V, const index_t _defect_length, const real_t cr) {
    do_init_defect(L0, L1, L2, L3, V);
    this->defect_length = _defect_length;
    set_defect(cr);
    Kokkos::fence();
    }

    template <class RNG>
    void do_init(const index_t L0, const index_t L1, 
                 const index_t L2, const index_t L3, 
                 GaugeField<Nd,Nc> &V, RNG &rng, const real_t delta) {
      Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
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
      tune_and_launch_for<Nd>(IndexArray<Nd>{0, 0, 0, 0}, IndexArray<Nd>{L0, L1, L2, L3},
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

    // Sets the defect value, this will need to be changed if the defect_length/position should become dynamic
    // with lengths on the order of 2 to 5, will this need parallelization? 
    void set_defect(real_t cr){
      #pragma unroll //does pragma unroll even make sense here?
      for (index_t i1; i1<defect_length; ++i1){
        #pragma unroll
        for (index_t i2; i2<defect_length;++i2){
          #pragma unroll
          for (index_t i3; i3<defect_length; ++i3){
            defectField(0,i1,i2,i3,0) = cr;
          }
        }
      }
    }

    GaugeField<Nd,Nc> field;
    LinkScalarField<Nd> defectField;
    index_t defect_length;

    // define accessors for the field
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const index_t i, const index_t j, const index_t k, const index_t l, const int mu) const {
      return field(i,j,k,l,mu)*defectField(i,j,k,l,mu);
    }
  
    KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> & operator()(const index_t i, const index_t j, const index_t k, const index_t l, const int mu) {
      return field(i,j,k,l,mu)*defectField(i,j,k,l,mu);
    }
    
  };

}
