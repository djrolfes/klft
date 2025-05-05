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

namespace klft
{

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField<Nd,Nc> operator*=(devicePTBCGaugeField<Nd,Nc> &a, const devicePTBCGaugeField<Nd,Nc> &b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,i2,i3,mu) *= b(i0,i1,i2,i3,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField3D<Nd,Nc> operator*=(devicePTBCGaugeField3D<Nd,Nc> &a, const devicePTBCGaugeField3D<Nd,Nc> &b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField3D",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,i2,mu) *= b(i0,i1,i2,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField2D<Nd,Nc> operator*=(devicePTBCGaugeField2D<Nd,Nc> &a, const devicePTBCGaugeField2D<Nd,Nc> &b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField2D",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,mu) *= b(i0,i1,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField<Nd,Nc> operator*=(devicePTBCGaugeField<Nd,Nc> &a, const deviceGaugeField<Nd,Nc> &b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField*deviceGaugeField",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,i2,i3,mu) *= b(i0,i1,i2,i3,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField3D<Nd,Nc> operator*=(devicePTBCGaugeField3D<Nd,Nc> &a, const deviceGaugeField3D<Nd,Nc> &b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField3D*deviceGaugeField3D",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,i2,mu) *= b(i0,i1,i2,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField2D<Nd,Nc> operator*=(devicePTBCGaugeField2D<Nd,Nc> &a, const deviceGaugeField2D<Nd,Nc> &b){
    assert(a.dimensions == b.dimensions);
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField2D*deviceGaugeField2D",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,mu) *= b(i0,i1,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField<Nd,Nc> operator*=(devicePTBCGaugeField<Nd,Nc> &a, const constGaugeField<Nd,Nc> &b){
    assert(a.field.layout() == b.layout());
    assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField*constGaugeField",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,i2,i3,mu) *= b(i0,i1,i2,i3,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField3D<Nd,Nc> operator*=(devicePTBCGaugeField3D<Nd,Nc> &a, const constGaugeField3D<Nd,Nc> &b){
    assert(a.field.layout() == b.layout());
    assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField3D*constGaugeField3D",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,i2,mu) *= b(i0,i1,i2,mu);
        }
      });
  }

  template<size_t Nd, size_t Nc>
  devicePTBCGaugeField2D<Nd,Nc> operator*=(devicePTBCGaugeField2D<Nd,Nc> &a, const constGaugeField2D<Nd,Nc> &b){
    assert(a.field.layout() == b.layout());
    assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
    tune_and_launch_for<Nd>("operator*=_devicePTBCGaugeField2D*constGaugeField2D",IndexArray<Nd>{0}, a.dimensions,
      KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
        #pragma unroll
        for (index_t mu = 0; mu < Nd; ++mu){
          a(i0,i1,mu) *= b(i0,i1,mu);
        }
      });
  }

    // define multiplication operations between deviceGaugeField and other fields.
    template<size_t Nd, size_t Nc>
    deviceGaugeField<Nd,Nc> operator*=(deviceGaugeField<Nd,Nc> &a, const deviceGaugeField<Nd,Nc> &b){
      assert(a.dimensions == b.dimensions);
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,i2,i3,mu) *= b(i0,i1,i2,i3,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField3D<Nd,Nc> operator*=(deviceGaugeField3D<Nd,Nc> &a, const deviceGaugeField3D<Nd,Nc> &b){
      assert(a.dimensions == b.dimensions);
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField3D",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,i2,mu) *= b(i0,i1,i2,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField2D<Nd,Nc> operator*=(deviceGaugeField2D<Nd,Nc> &a, const deviceGaugeField2D<Nd,Nc> &b){
      assert(a.dimensions == b.dimensions);
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField2D",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,mu) *= b(i0,i1,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField<Nd,Nc> operator*=(deviceGaugeField<Nd,Nc> &a, const devicePTBCGaugeField<Nd,Nc> &b){
      assert(a.dimensions == b.dimensions);
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField*devicePTBCGaugeField",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,i2,i3,mu) *= b(i0,i1,i2,i3,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField3D<Nd,Nc> operator*=(deviceGaugeField3D<Nd,Nc> &a, const devicePTBCGaugeField3D<Nd,Nc> &b){
      assert(a.dimensions == b.dimensions);
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField3D*devicePTBCGaugeField3D",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,i2,mu) *= b(i0,i1,i2,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField2D<Nd,Nc> operator*=(deviceGaugeField2D<Nd,Nc> &a, const devicePTBCGaugeField2D<Nd,Nc> &b){
      assert(a.dimensions == b.dimensions);
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField2D*_devicePTBCGaugeField2D",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,mu) *= b(i0,i1,mu);
          }
        });
    }
  
    // for constGaugeField, the type directly represents a View, so access/asserts need to be ammended.
    template<size_t Nd, size_t Nc>
    deviceGaugeField<Nd,Nc> operator*=(deviceGaugeField<Nd,Nc> &a, const constGaugeField<Nd,Nc> &b){
      assert(a.field.layout() == b.layout());
      assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField3D*constGaugeField",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2, const index_t i3) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,i2,i3,mu) *= b(i0,i1,i2,i3,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField3D<Nd,Nc> operator*=(deviceGaugeField3D<Nd,Nc> &a, const constGaugeField3D<Nd,Nc> &b){
      assert(a.field.layout() == b.layout());
      assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField3D*constGaugeField3D",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,i2,mu) *= b(i0,i1,i2,mu);
          }
        });
    }
  
    template<size_t Nd, size_t Nc>
    deviceGaugeField2D<Nd,Nc> operator*=(deviceGaugeField2D<Nd,Nc> &a, const constGaugeField2D<Nd,Nc> &b){
      assert(a.field.layout() == b.layout());
      assert(a.field.memory_space == b.memory_space); //only allow device-device and host-host operations
      tune_and_launch_for<Nd>("operator*=_deviceGaugeField2D*constGaugeField2D",IndexArray<Nd>{0}, a.dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          #pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu){
            a(i0,i1,mu) *= b(i0,i1,mu);
          }
        });
  
      }

  // calculate staple per site and store in another gauge fieldDeviceGaugeFieldType
  template <size_t Nd, size_t Nc, GaugeFieldKind k = GaugeFieldKind::Standard>
  const constGaugeField<Nd,Nc> stapleField(const DeviceGaugeFieldType<Nd ,Nc, k>::type g_in) {
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