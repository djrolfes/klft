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
// applys Composid Operator Mdagger= DDdagger*s_in
struct TagM {};
// applys Composid Operator Mdagger= DdaggerD*s_in
struct TagMdagger {};
}  // namespace Tags

template <typename _Derived,
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

  using Derived = _Derived;
  // Define Tags for template dispatch:
  using SpinorFieldType = typename DSpinorFieldType::type;
  using GaugeFieldType = typename DGaugeFieldType::type;

 public:
  ~DiracOperator() = default;

  SpinorFieldType apply(Tags::TagD, const SpinorFieldType& s_in) {
    // Initialize the input field
    this->s_in = s_in;
    // Initialize the output field
    this->s_out = SpinorFieldType(params.dimensions, complex_t(0.0, 0.0));
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagD>(typeid(Derived).name(),
                                          IndexArray<rank>{}, params.dimensions,
                                          static_cast<_Derived&>(*this));
    Kokkos::fence();
    return s_out;
  }

  // keep specilized apply functions for convience

  SpinorFieldType apply(Tags::TagDdagger, const SpinorFieldType& s_in) {
    this->s_in = s_in;
    // Initialize the output field
    this->s_out = SpinorFieldType(params.dimensions, complex_t(0.0, 0.0));
    // Apply the operator
    tune_and_launch_for<rank, Tags::TagDdagger>(
        typeid(Derived).name(), IndexArray<rank>{}, params.dimensions,
        static_cast<_Derived&>(*this));
    Kokkos::fence();
    return s_out;
  }
  void applyD_inplace(const SpinorFieldType& s_in, SpinorFieldType& s_out) {
    this->s_in = s_in;
    this->s_out = s_out;
    tune_and_launch_for<rank, Tags::TagD>(typeid(Derived).name(),
                                          IndexArray<rank>{}, params.dimensions,
                                          static_cast<_Derived&>(*this));
    Kokkos::fence();
  }
  void applyDdagger_inplace(const SpinorFieldType& s_in,
                            SpinorFieldType& s_out) {
    this->s_in = s_in;
    this->s_out = s_out;
    tune_and_launch_for<rank, Tags::TagDdagger>(
        typeid(Derived).name(), IndexArray<rank>{}, params.dimensions,
        static_cast<_Derived&>(*this));
    Kokkos::fence();
  }
  // applys Composid Operator M= DDdagger*s_in

  SpinorFieldType apply(Tags::TagM, const SpinorFieldType& s_in) {
    SpinorFieldType temp = this->apply(Tags::TagDdagger{}, s_in);
    return this->apply(Tags::TagD{}, temp);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply(Tags::TagMdagger, const SpinorFieldType& s_in) {
    SpinorFieldType temp = this->apply(Tags::TagD{}, s_in);
    return this->apply(Tags::TagDdagger{}, temp);
  }
  SpinorFieldType s_in;
  SpinorFieldType s_out;
  const GaugeFieldType g_in;
  const diracParams<rank, RepDim> params;

  DiracOperator(const GaugeFieldType& g_in,
                const diracParams<rank, RepDim>& params)
      : g_in(g_in), params(params) {}
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION SpinorFieldType
  apply(const SpinorFieldType& s_in) {
    // Apply the operator
    return this->apply(Tag{}, s_in);
  }

 protected:
  DiracOperator() = default;
};

template <typename DSpinorFieldType, typename DGaugeFieldType>
class WilsonDiracOperator
    : public DiracOperator<
          WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
          DSpinorFieldType,
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
      DiracOperator<WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                    DSpinorFieldType,
                    DGaugeFieldType>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 0, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 0, -1,
          this->params.dimensions);

      auto XPlus = this->s_in(xp.first);
      auto temp1 = (this->params.gamma_id - this->params.gammas[mu]) * XPlus;
      auto temp2 = this->g_in(Idcs..., mu) * temp1;

      auto Xminus = this->s_in(xm.first);
      auto temp3 = (this->params.gamma_id + this->params.gammas[mu]) * Xminus;
      auto temp4 = conj(this->g_in(xm.first, mu)) * temp3;
      temp += ((this->params.kappa * xp.second) * temp2 +
               (this->params.kappa * xm.second) * temp4);
    }

    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 0, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 0, -1,
          this->params.dimensions);
      auto XPlus = this->s_in(xp.first);
      auto temp1 = (this->params.gamma_id + this->params.gammas[mu]) * XPlus;
      auto temp2 = this->g_in(Idcs..., mu) * temp1;

      auto Xminus = this->s_in(xm.first);
      auto temp3 = (this->params.gamma_id - this->params.gammas[mu]) * Xminus;
      auto temp4 = conj(this->g_in(xm.first, mu)) * temp3;
      temp += (this->params.kappa * xp.second) * temp2 +
              (this->params.kappa * xm.second) * temp4;
    }
    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
  }
};
// // Deduction guide
// template <typename GaugeType, typename ParamType>
// WilsonDiracOperator(const GaugeType&, const ParamType&)
//     -> WilsonDiracOperator<ParamType::rank, ParamType::Nc,
//     ParamType::RepDim>;

template <typename DSpinorFieldType, typename DGaugeFieldType>
class HWilsonDiracOperator
    : public DiracOperator<
          HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
          DSpinorFieldType,
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
      DiracOperator<HWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>,
                    DSpinorFieldType,
                    DGaugeFieldType>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 0, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, 0, -1,
          this->params.dimensions);

      temp += (this->params.gamma_id - this->params.gammas[mu]) * xp.second *
              (this->g_in(Idcs..., mu) * this->s_in(xp.first));
      temp += (this->params.gamma_id + this->params.gammas[mu]) * xm.second *
              (conj(this->g_in(xm.first, mu)) * this->s_in(xm.first));
    }

    this->s_out(Idcs...) =
        this->params.gamma5 * (this->s_in(Idcs...) - this->params.kappa * temp);
  }

  // only for testing porpose, not the real Ddagger operator
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    operator()(typename Base::TagD(), Idcs...);
  }
};
// // Deduction guide
// template <typename GaugeType, typename ParamType>
// HWilsonDiracOperator(const GaugeType&, const ParamType&)
//     -> HWilsonDiracOperator<ParamType::rank, ParamType::Nc,
//     ParamType::RepDim>;

}  // namespace klft