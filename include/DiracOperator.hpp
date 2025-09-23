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

// this file defines various versions of the Wilson-Dirac (WD) operator, in
// lattice units following Gattringer2010 (5.55) f. and absorbing the constant C
// into the field definition
#pragma once
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "IndexHelper.hpp"
#include "Spinor.hpp"
#include "SpinorFieldLinAlg.hpp"

namespace klft {
// Base Class for Single Dirac Operator
namespace Tags {
struct TagD {};
struct TagDdagger {};
// applys Composid Operator M= DDdagger*s_in
struct TagDDdagger {};
// applys Composid Operator Mdagger= DdaggerD*s_in
// If spinorfield are Checkerboard the operator S_e is used
struct TagDdaggerD {};
struct TagHoe {};
struct TagHeo {};
struct TagSe {};
struct TagSo {};
}  // namespace Tags

template <template <typename, typename> class _Derived,
          typename DSpinorFieldType,
          typename DGaugeFieldType>
class DiracOperator {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout;

  using Derived = _Derived<DSpinorFieldType, DGaugeFieldType>;
  // Define Tags for template dispatch:
  using SpinorFieldType = typename DSpinorFieldType::type;
  using GaugeFieldType = typename DGaugeFieldType::type;

 public:
  DiracOperator(const GaugeFieldType& g_in, const diracParams& params)
      : g_in(g_in), params(params) {}
  ~DiracOperator() = default;
  // Define callabale apply functions
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION SpinorFieldType
  apply(const SpinorFieldType& s_in) {
    this->s_in = s_in;
    this->s_out = SpinorFieldType(s_in.dimensions, complex_t(0.0, 0.0));
    // Apply the operator
    if constexpr (std::is_same_v<Tag, Tags::TagHeo> or
                  std::is_same_v<Tag, Tags::TagHoe>) {
      // forward to Derived class only for TagHeo
      return static_cast<Derived&>(*this).apply_(Tag{});
    } else {
      return this->apply_(Tag{});
    }
  }

  /// @brief applys the DiracOperator to the field s_in and stores the result in
  /// s_out
  /// @tparam Tag Can either be TagD or TagDdagger or TagDDdagger or TagDdaggerD
  /// @param s_in
  /// @param s_out
  /// @return Nothing, result is stored in s_out
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_out) {
    this->s_in = s_in;
    this->s_out = s_out;
    if constexpr (std::is_same_v<Tag, Tags::TagHeo> or
                  std::is_same_v<Tag, Tags::TagHoe>) {
      // forward to Derived class only for TagHeo
      static_cast<Derived&>(*this).apply_(Tag{});
    } else {
      this->apply_(Tag{});
    }
  }

  // Special overload t reduce allocations in solver further, only works with
  // composed operators

  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_temp,
                                         const SpinorFieldType& s_out) {
    this->s_in = s_in;
    this->s_out = s_temp;

    // Apply the operator
    this->apply_(Tag{}, s_out);
  }

 protected:
  SpinorFieldType apply_(Tags::TagD) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagD>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return s_out;
  }

  SpinorFieldType apply_(Tags::TagDdagger) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagDdagger>(
        typeid(Derived).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return s_out;
  }
  // applys Composid Operator M= DDdagger*s_in

  SpinorFieldType apply_(Tags::TagDDdagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDDdagger{}, cached_out);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply_(Tags::TagDdaggerD) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDdaggerD{}, cached_out);
  }
  SpinorFieldType apply_(Tags::TagSe) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSe{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagSe, const SpinorFieldType& s_out) {
    auto cached_out = this->s_out;

    return static_cast<Derived&>(*this).apply_(Tags::TagSe{}, s_out);
  }
  SpinorFieldType apply_(Tags::TagSo) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSo{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagSo, const SpinorFieldType& s_out) {
    auto cached_out = this->s_out;
    static_cast<Derived&>(*this).apply_(Tags::TagSo{}, s_out);
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
      // printf("Iam a CheckerboardLayout\n");
      static_cast<Derived&>(*this).apply_(Tags::TagDDdagger{}, s_out);
      // auto temp_field = this->s_in;
      // apply<Tags::TagHoe>(this->s_in, this->s_out);
      // this->s_in = this->s_out;
      // this->s_out = s_out;
      // apply<Tags::TagHeo>(this->s_in, this->s_out);
      // axpy<DSpinorFieldType>(-1, this->s_out, temp_field, this->s_out);
      return this->s_out;
    }

    this->apply_(Tags::TagDdagger{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply_(Tags::TagD{});
  }

  SpinorFieldType apply_(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
      static_cast<Derived&>(*this).apply_(Tags::TagDdaggerD{}, s_out);
      // printf("Iam a CheckerboardLayout\n");
      // auto temp_field = this->s_in;
      // this->apply_(Tags::TagD{});
      // this->s_in = this->s_out;
      // this->s_out = s_out;
      // this->apply_(Tags::TagDdagger{});
      // axpy<DSpinorFieldType>(-1, this->s_out, temp_field, this->s_out);
      return this->s_out;
    }
    this->apply_(Tags::TagD{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply_(Tags::TagDdagger{});
  }

 public:
  SpinorFieldType s_in;
  SpinorFieldType s_out;
  const GaugeFieldType g_in;
  const diracParams params;

 protected:
  DiracOperator() = default;
};
// intermidiate base class for EO Precon. Dirac Operators
template <template <typename, typename> class _Derived,
          typename DSpinorFieldType,
          typename DGaugeFieldType>
class EODiracOperator
    : public DiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType> {
  using SpinorFieldType = typename DSpinorFieldType::type;
  using InterDervied = _Derived<DSpinorFieldType, DGaugeFieldType>;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;

 public:
  SpinorFieldType s_in_same_parity;
  SpinorFieldType temp;

  struct Tag1minusHeo {};
  struct Tag1minusHoe {};
  int test = 0;
  using DiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType>::
      DiracOperator;

  SpinorFieldType apply_(Tags::TagHeo) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));
    tune_and_launch_for<rank, Tags::TagHeo>(
        typeid(InterDervied).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<InterDervied&>(*this));
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagHoe) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));

    tune_and_launch_for<rank, Tags::TagHoe>(
        typeid(InterDervied).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<InterDervied&>(*this));
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagSe, const SpinorFieldType& s_out) {
    // printf("iam a EO Se\n");
    // printf("%i", this->test);
    this->apply_(Tags::TagHoe{});

    this->s_in_same_parity = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    tune_and_launch_for<rank, Tag1minusHeo>(
        typeid(InterDervied).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<InterDervied&>(*this));

    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagSo, const SpinorFieldType& s_out) {
    this->apply_(Tags::TagHeo{});
    this->s_in_same_parity = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    tune_and_launch_for<rank, Tag1minusHoe>(
        typeid(InterDervied).name(), IndexArray<rank>{}, this->s_in.dimensions,
        static_cast<InterDervied&>(*this));

    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    if (!temp.field.is_allocated()) {
      this->temp = SpinorFieldType(this->s_in.dimensions, 0);
      printf("instatinated temp\n");
    }

    auto cached_s_out = this->s_out;
    apply_(Tags::TagSe{}, temp);
    this->s_in = temp;
    this->s_out = cached_s_out;
    // out is not correctly set for first Heo application
    apply_(Tags::TagSe{}, s_out);
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    apply_(Tags::TagDDdagger{}, s_out);
    return this->s_out;
  }
};

}  // namespace klft