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

namespace klft {
// Base Class for Single Dirac Operator
namespace Tags {
struct TagD {};
struct TagDdagger {};
// applys Composid Operator M= DDdagger*s_in
struct TagDDdagger {};
// applys Composid Operator Mdagger= DdaggerD*s_in
struct TagDdaggerD {};
}  // namespace Tags

template <template <typename, typename> class _Derived,
          typename DSpinorFieldType, typename DGaugeFieldType>
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
    this->dimensions = s_in.dimensions;
    this->s_out = SpinorFieldType(this->dimensions, complex_t(0.0, 0.0));
    // Apply the operator
    return this->apply(Tag{});
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
    this->dimensions = s_in.dimensions;
    this->s_out = s_out;
    // Apply the operator
    this->apply(Tag{});
  }

  // Special overload t reduce allocations in solver further, only works with
  // composed operators

  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_temp,
                                         const SpinorFieldType& s_out) {
    this->dimensions = s_in.dimensions;
    this->s_in = s_in;
    this->s_out = s_temp;
    // Apply the operator
    this->apply(Tag{}, s_out);
  }

 private:
  SpinorFieldType apply(Tags::TagD) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagD>(typeid(Derived).name(),
                                          IndexArray<rank>{}, this->dimensions,
                                          static_cast<Derived&>(*this));
    Kokkos::fence();
    return s_out;
  }

  SpinorFieldType apply(Tags::TagDdagger) {
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagDdagger>(
        typeid(Derived).name(), IndexArray<rank>{}, this->dimensions,
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return s_out;
  }
  // applys Composid Operator M= DDdagger*s_in

  SpinorFieldType apply(Tags::TagDDdagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->dimensions, complex_t(0.0, 0.0));

    return this->apply(Tags::TagDDdagger{}, cached_out);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply(Tags::TagDdaggerD) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->dimensions, complex_t(0.0, 0.0));

    return this->apply(Tags::TagDdaggerD{}, cached_out);
  }
  SpinorFieldType apply(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    this->apply(Tags::TagDdagger{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply(Tags::TagD{});
  }

  SpinorFieldType apply(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    this->apply(Tags::TagD{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply(Tags::TagDdagger{});
  }

 public:
  SpinorFieldType s_in;
  SpinorFieldType s_out;
  const GaugeFieldType g_in;
  const diracParams params;
  Kokkos::Array<index_t, rank> dimensions;

 protected:
  DiracOperator() = default;
};

template <typename DSpinorFieldType, typename DGaugeFieldType>
class WilsonDiracOperator
    : public DiracOperator<WilsonDiracOperator, DSpinorFieldType,
                           DGaugeFieldType> {
 public:
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;

  ~WilsonDiracOperator() = default;
  using Base =
      DiracOperator<WilsonDiracOperator, DSpinorFieldType, DGaugeFieldType>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, -1, this->s_in(xp.first));

      auto temp2 =
          conj(this->g_in(xm.first, mu)) * project(mu, 1, this->s_in(xm.first));
      temp += reconstruct(mu, -1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, 1, (this->params.kappa * xm.second) * temp2);
    }

    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};

#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, 1, this->s_in(xp.first));

      //
      auto temp2 = conj(this->g_in(xm.first, mu)) *
                   project(mu, -1, this->s_in(xm.first));
      temp += reconstruct(mu, 1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, -1, (this->params.kappa * xm.second) * temp2);
    }
    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
  }
};

template <typename DSpinorFieldType, typename DGaugeFieldType>
class HWilsonDiracOperator
    : public DiracOperator<HWilsonDiracOperator, DSpinorFieldType,
                           DGaugeFieldType> {
 public:
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;

  ~HWilsonDiracOperator() = default;
  using Base =
      DiracOperator<HWilsonDiracOperator, DSpinorFieldType, DGaugeFieldType>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, 1, this->s_in(xp.first));

      //
      auto temp2 = conj(this->g_in(xm.first, mu)) *
                   project(mu, -1, this->s_in(xm.first));
      temp += reconstruct(mu, 1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, -1, (this->params.kappa * xm.second) * temp2);
    }

    this->s_out(Idcs...) = gamma5(this->s_in(Idcs...) - temp);
  }

  // only for testing porpose, not the real Ddagger operator
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    operator()(typename Tags::TagD(), Idcs...);
  }
};

}  // namespace klft