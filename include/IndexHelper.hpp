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

// this file defines helper functions for index manipulation

#pragma once
#include "GLOBAL.hpp"

namespace klft
{

  // return x + shift  mu
  template <size_t rank, int shift, typename indexType>
  constexpr
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::Array<index_t,rank> shift_index_plus(const Kokkos::Array<indexType,rank> &idx,
                                          const index_t mu,
                                          const IndexArray<rank> &dimensions) {
    // make sure mu makes sense
    assert(mu < rank && mu >= 0);
    Kokkos::Array<index_t,rank> new_idx;
    #pragma unroll
    for (index_t i = 0; i < rank; ++i) {
      new_idx[i] = static_cast<index_t>(idx[i]);
    }
    new_idx[mu] = (idx[mu] + shift) % dimensions[mu];
    return new_idx;
  }

  // return x - shift mu
  template <size_t rank, int shift, typename indexType>
  constexpr
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::Array<index_t,rank> shift_index_minus(const Kokkos::Array<indexType,rank> &idx,
                                          const index_t mu,
                                          const IndexArray<rank> &dimensions) {
    // make sure mu makes sense
    assert(mu < rank && mu >= 0);
    Kokkos::Array<index_t,rank> new_idx;
    #pragma unroll
    for (index_t i = 0; i < rank; ++i) {
      new_idx[i] = static_cast<index_t>(idx[i]);
    }
    new_idx[mu] = (idx[mu] - shift + dimensions[mu]) % dimensions[mu];
    return new_idx;
  }

}