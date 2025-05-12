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

// utility functions for gauge fields
#pragma once
#include "GaugeField.hpp"
#include "PTBCGaugeField.hpp"
#include "SUN.hpp"

namespace klft
{

  // define multiplication between GaugeFields for now only define *= operators, 
  // as I have yet to decide on a heuristic for the output type when mixing different GaugeFieldKinds
  template<size_t Nd, size_t Nc, GaugeFieldKind K1, GaugeFieldKind K2>
  typename DeviceGaugeFieldType<Nd, Nc, K1>::type& 
  operator*=(
    typename DeviceGaugeFieldType<Nd, Nc, K1>::type &a, 
    typename DeviceGaugeFieldType<Nd, Nc, K2>::type const & b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc, k1>::type*DeviceGaugeFieldType<Nd, Nc, k2>::type",IndexArray<Nd>{0}, a.dimensions,
    KOKKOS_LAMBDA(auto... idxs){
      #pragma unroll
      for (index_t mu = 0;mu<Nd;++mu){
        a.field(idxs..., mu) *= b(idxs...., mu);
      }
    });
    return a;
  }

  template<size_t Nd, size_t Nc, GaugeFieldKind k>
  GaugeField<Nd, Nc>& operator*=(GaugeField<Nd, Nc> &a, 
    typename DeviceGaugeFieldType<Nd, Nc, k>::type const &b){
    assert(a.layout() == b.field.layout());
    assert(a.memory_space == b.field.memory_space); //only allow device-device and host-host operations
    tune_and_launch_for<Nd>("operator*=GaugeField*DeviceGaugeFieldType<Nd, Nc, k>::type",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(auto... idxs){
        #pragma unroll
        for (index_t mu = 0;mu<Nd;++mu){
          a(idxs..., mu) *= b(idxs...., mu);
        }
      });
    return a;
  }

  template<size_t Nd, size_t Nc, GaugeFieldKind k>
  typename DeviceGaugeFieldType<Nd, Nc, k>::type& operator*=(
    typename DeviceGaugeFieldType<Nd, Nc, k>::type &a, 
    GaugeField<Nd, Nc> const &b){
    assert(a.field.layout() == b.layout());
    assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
    tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc, k>::type*GaugeField",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(auto... idxs){
        #pragma unroll
        for (index_t mu = 0;mu<Nd;++mu){
          a.field(idxs..., mu) *= b(idxs...., mu);
        }
      });
    return a;
  }

  template<size_t Nd, size_t Nc, GaugeFieldKind k>
  typename DeviceGaugeFieldType<Nd, Nc, k>::type& operator*=(
    typename DeviceGaugeFieldType<Nd, Nc, k>::type &a, 
    constGaugeField<Nd, Nc> const &b){
    assert(a.field.layout() == b.layout());
    assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
    tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc, k>::type*constGaugeField",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(auto... idxs){
        #pragma unroll
        for (index_t mu = 0;mu<Nd;++mu){
          a.field(idxs..., mu) *= b(idxs...., mu);
        }
      });
    return a;
  }
  // should the multiplication between GaugeField<...> types also be defined?

  // now define the multiplication with scalars
  template<size_t Nd, size_t Nc, GaugeFieldKind K1>
  typename DeviceGaugeFieldType<Nd, Nc, K1>::type& 
  operator*=(
    typename DeviceGaugeFieldType<Nd, Nc, K1>::type &a, 
    real_t const b){
    tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc, k1>::type*Scalar",IndexArray<Nd>{0}, a.dimensions,
    KOKKOS_LAMBDA(auto... idxs){
      #pragma unroll
      for (index_t mu = 0;mu<Nd;++mu){
        a.field(idxs..., mu) *= b;
      }
    });
    return a;
  }

  template<size_t Nd, size_t Nc>
  GaugeField<Nd, Nc>& operator*=(GaugeField<Nd, Nc> &a, 
    real_t const b){
    tune_and_launch_for<Nd>("operator*=GaugeField*Scalar",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(auto... idxs){
        #pragma unroll
        for (index_t mu = 0;mu<Nd;++mu){
          a(idxs..., mu) *= b;
        }
      });
    return a;
  }



  // function to conjugate a given DeviceGaugeField in place.
  template <size_t Nd, size_t Nc, GaugeFieldKind K = GaugeFieldKind::Standard>
  void conj_field(typename DeviceGaugeFieldType<Nd, Nc, K>::type& field) {
    tune_and_launch_for<Nd>("conj_field",IndexArray<Nd>{0}, field.dimensions,
    KOKKOS_LAMBDA(auto... idxs){
      #pragma unroll
      for (index_t mu = 0; mu <Nd; ++mu){
        field.field(idxs..., mu) = conj(field(idxs..., mu));
      }
    });
    Kokkos::fence();
  }


  // calculate staple per site and store in another gauge fieldDeviceGaugeFieldType
  template <size_t Nd, size_t Nc, GaugeFieldKind k = GaugeFieldKind::Standard>
  const constGaugeField<Nd,Nc> stapleField(typename DeviceGaugeFieldType<Nd ,Nc, k>::type const & g_in) {
    // initialize the output field
    using FieldT = DeviceGaugeFieldType<Nd ,Nc, k>::type
    switch (Nd)
    {
    case 4:
    FieldT g_out(g_in.field.extent(0), g_in.field.extent(1), 
      g_in.field.extent(2), g_in.field.extent(3), complex_t(0.0, 0.0));
      break;
    case 3:
    FieldT g_out(g_in.field.extent(0), g_in.field.extent(1), 
      g_in.field.extent(2), complex_t(0.0, 0.0));
      break;
    case 2:
    FieldT g_out(g_in.field.extent(0), g_in.field.extent(1), complex_t(0.0, 0.0));
    break;
    default:
      assert(1==2); //TODO: do a decent error
      break;
    }
    
    // get the start and end indices
    const auto & dimensions = g_in.field.layout().dimension;
    IndexArray<Nd> start;
    IndexArray<Nd> end;
    for (index_t i = 0; i < Nd; ++i) {
      start[i] = 0;
      end[i] = dimensions[i];
    }

    // It would be trivial to add a stapleField return into each DeviceGaugeFieldType, 
    // though as already done with .staple, shouldn't the definition and calculations be seperated?
    switch (Nd)
    {
    case 4:
    tune_and_launch_for<Nd>("stapleField_GaugeField", start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        
        for (index_t mu = 0; mu<Nd, ++mu){
          g_out(i0,i1,i2,i3,mu) = g_in.staple(i0,i1,i2,i3,mu);
        }
      });
      break;
    case 3:
    tune_and_launch_for<Nd>("stapleField_GaugeField3D", start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
        
        for (index_t mu = 0; mu<Nd, ++mu){
          g_out(i0,i1,i2,mu) = g_in.staple(i0,i1,i2,mu);
        }
      });
      break;
    case 2:
    tune_and_launch_for<Nd>("stapleField_GaugeField3D", start, end,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
        
        for (index_t mu = 0; mu<Nd, ++mu){
          g_out(i0,i1,mu) = g_in.staple(i0,i1,mu);
        }
      });
    break;
    }
    Kokkos::fence();
    // return the output field
    return constGaugeField<Nd,Nc>(g_out.field);
  }

}