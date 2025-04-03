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

// Define the global types and views for Kokkos Lattice Field Theory (KLFT)
// This file contains the definitions for the types used in KLFT, including
// the real and complex types, gauge field types, and field view types.
// It also includes the definitions for the policies used in Kokkos parallel
// programming.


#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace klft
{

  // defining a compile time verbosity here
  // should later be changed to a runtime option
  #ifndef KLFT_VERBOSITY
  #define KLFT_VERBOSITY 5
  #endif

  // define the global types
  // is this a good idea?
  // if we want different fields with different precision (e.g. coarse grid)
  // might be a lot of work to redefine, but on the other hand, if we want to
  // be precision agnostic, everything has to be templated with given precision
  // let's start like this and worry about mixed precision later
  using real_t = double;
  using complex_t = Kokkos::complex<real_t>;

  // use int for the index type
  using index_t = int;

  // define index_arrays
  template <size_t rank>
  using IndexArray = Kokkos::Array<index_t,rank>;

  // define groups for gauge fields
  template <size_t Nc>
  using SUN = Kokkos::Array<Kokkos::Array<complex_t,Nc>,Nc>;

  // define field view types
  // by default all views are 4D
  // some dimensions are set to 1 for lower dimensions
  // I'm still not sure if this is the best way to do it
  // Nd here is templated, but for a 4D gauge field,
  // shouldn't Nd always be 4?
  // Nc is the number of colors
  template <size_t Nd, size_t Nc>
  using GaugeField = Kokkos::View<SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

  template <size_t Nc>
  using SUNField = Kokkos::View<SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  using Field = Kokkos::View<complex_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  using ScalarField = Kokkos::View<real_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  // define corresponding constant fields
  #if defined ( KOKKOS_ENABLE_CUDA )

  template <size_t Nd, size_t Nc>
  using constGaugeField = Kokkos::View<const SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  template <size_t Nc>
  using constSUNField = Kokkos::View<const SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  using constField = Kokkos::View<const complex_t****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  using constScalarField = Kokkos::View<const real_t****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

  #else

  template <size_t Nd, size_t Nc>
  using constGaugeField = Kokkos::View<const SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

  template <size_t Nc>
  using constSUNField = Kokkos::View<const SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  using constField = Kokkos::View<const complex_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  using constScalarField = Kokkos::View<const real_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

  #endif

  // define policy as mdrange
  template <size_t rank>
  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

  // special case for 1D
  using Policy1D = Kokkos::RangePolicy<>;

}