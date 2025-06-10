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

// this is a helper file to define the field types based on dimension
// for the gauge fields here we assume that Nd = rank (dimensionality)
// this would work for most use cases. But if you have a different
// dimensionality from Nd, you can not use the definitions here

#pragma once
#include "Field.hpp"
#include "GaugeField.hpp"
#include "PTBCGaugeField.hpp"
#include "SUNField.hpp"
#include "ScalarField.hpp"
#include <cstddef>

namespace klft {
// define GaugeFieldKinds
enum class GaugeFieldKind { Standard, PTBC };

// define a function to get the gauge field type based on the rank,
// with the default Field being the default GaugeField
template <size_t rank, size_t Nc, GaugeFieldKind k = GaugeFieldKind::Standard>
struct DeviceGaugeFieldType;

// now define the specializations
template <size_t Nc>
struct DeviceGaugeFieldType<2, Nc, GaugeFieldKind::Standard> {
  using type = deviceGaugeField2D<2, Nc>;
};

template <size_t Nc>
struct DeviceGaugeFieldType<3, Nc, GaugeFieldKind::Standard> {
  using type = deviceGaugeField3D<3, Nc>;
};

template <size_t Nc>
struct DeviceGaugeFieldType<4, Nc, GaugeFieldKind::Standard> {
  using type = deviceGaugeField<4, Nc>;
};

// now do the same for the PTBC gauge field types
template <size_t Nc> struct DeviceGaugeFieldType<4, Nc, GaugeFieldKind::PTBC> {
  using type = devicePTBCGaugeField<4, Nc>;
};

template <size_t Nc> struct DeviceGaugeFieldType<3, Nc, GaugeFieldKind::PTBC> {
  using type = devicePTBCGaugeField3D<3, Nc>;
};

template <size_t Nc> struct DeviceGaugeFieldType<2, Nc, GaugeFieldKind::PTBC> {
  using type = devicePTBCGaugeField2D<2, Nc>;
};

// define Traits to extract the rank, Nc and GaugeFieldKind at a later point
template <typename T> struct DeviceGaugeFieldTypeTraits;

template <size_t _rank, size_t _Nc, GaugeFieldKind _k>
struct DeviceGaugeFieldTypeTraits<DeviceGaugeFieldType<_rank, _Nc, _k>> {
  static constexpr size_t Rank = _rank;
  static constexpr size_t Nc = _Nc;
  static constexpr GaugeFieldKind Kind = _k;
};

// define the same thing for SUN fields
template <size_t rank, size_t Nc> struct DeviceSUNFieldType;

template <size_t Nc> struct DeviceSUNFieldType<2, Nc> {
  using type = deviceSUNField2D<Nc>;
};

template <size_t Nc> struct DeviceSUNFieldType<3, Nc> {
  using type = deviceSUNField3D<Nc>;
};

template <size_t Nc> struct DeviceSUNFieldType<4, Nc> {
  using type = deviceSUNField<Nc>;
};

// repeat for field
template <size_t rank> struct DeviceFieldType;

template <> struct DeviceFieldType<2> {
  using type = deviceField2D;
};

template <> struct DeviceFieldType<3> {
  using type = deviceField3D;
};

template <> struct DeviceFieldType<4> {
  using type = deviceField;
};

// define the same for the scalar fields
template <size_t rank> struct DeviceScalarFieldType;

template <> struct DeviceScalarFieldType<2> {
  using type = deviceScalarField2D;
};

template <> struct DeviceScalarFieldType<3> {
  using type = deviceScalarField3D;
};

template <> struct DeviceScalarFieldType<4> {
  using type = deviceScalarField;
};

// add the same for scalar fields here when needed
} // namespace klft
