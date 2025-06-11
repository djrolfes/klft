
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

// this file defines functions to calculate
// Wilson Loops for different gauge fields
// and different gauge groups

#pragma once
#include "FieldTypeHelper.hpp"
#include "IndexHelper.hpp"

namespace klft
{


template <size_t rank, size_t Nc>
struct density_E{

const DeviceGaugeFieldType<rank, Nc> gin;

	void density_E(DeviceGaugeFieldType<rank, Nc> _gin): gin = _gin {}
};


}
