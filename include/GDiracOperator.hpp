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
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "IndexHelper.hpp"
#include "Spinor.hpp"

namespace klft {

template <size_t _rank, size_t _Nc, size_t _RepDim>
struct diracParameters {
  static constexpr size_t rank = _rank;
  static constexpr size_t Nc = _Nc;
  static constexpr size_t RepDim = _RepDim;
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const GammaMat<RepDim> gamma5;
  const real_t kappa;
  const IndexArray<rank> dimensions;
  diracParameters(const IndexArray<rank> _dimensions,
                  const VecGammaMatrix& _gammas,
                  const GammaMat<RepDim>& _gamma5, const real_t& _kappa)
      : dimensions(_dimensions),
        gammas(_gammas),
        gamma5(_gamma5),
        kappa(_kappa) {}
};

template <typename _Derived, size_t rank, size_t Nc, size_t RepDim>
class DiracOperator : public std::enable_shared_from_this<
                          DiracOperator<_Derived, rank, Nc, RepDim>> {
  using Derived = _Derived;
  // Define Tags for template dispatch:
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;

 public:
  struct TagD {};
  struct TagDdagger {};
  ~DiracOperator() = default;

  SpinorFieldType applyD(const SpinorFieldType& s_in) {
    // Initialize the input field
    this->s_in = s_in;
    // Initialize the output field
    this->s_out = SpinorFieldType(params.dimensions, complex_t(0.0, 0.0));
    // Apply the operator
    tune_and_launch_for<rank, TagD>(typeid(Derived).name(), IndexArray<rank>{},
                                    params.dimensions,
                                    static_cast<_Derived&>(*this));
    return s_out;
  }
  SpinorFieldType applyDdagger(const SpinorFieldType& s_in) {
    this->s_in = s_in;
    // Initialize the output field
    this->s_out = SpinorFieldType(params.dimensions, complex_t(0.0, 0.0));
    // Apply the operator
    tune_and_launch_for<rank, TagDdagger>(typeid(Derived).name(),
                                          IndexArray<rank>{}, params.dimensions,
                                          static_cast<_Derived&>(*this));
    return s_out;
  }
  SpinorFieldType s_in;
  SpinorFieldType s_out;
  const GaugeFieldType g_in;
  const diracParameters<rank, Nc, RepDim> params;

  DiracOperator(const GaugeFieldType& g_in,
                const diracParameters<rank, Nc, RepDim>& params)
      : g_in(g_in), params(params) {}

 protected:
  DiracOperator() = default;
};

template <size_t _rank, size_t _Nc, size_t _RepDim>
class WilsonDiracOperator
    : public DiracOperator<WilsonDiracOperator<_rank, _Nc, _RepDim>, _rank, _Nc,
                           _RepDim> {
 public:
  constexpr static size_t Nc = _Nc;
  constexpr static size_t RepDim = _RepDim;
  constexpr static size_t rank = _rank;

  ~WilsonDiracOperator() = default;
  using Base =
      DiracOperator<WilsonDiracOperator<rank, Nc, RepDim>, rank, Nc, RepDim>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
#pragma unroll
    for (size_t mu = 0; mu < _rank; ++mu) {
      auto xm = shift_index_minus_bc<_rank, size_t>(
          Kokkos::Array<size_t, _rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<_rank, size_t>(
          Kokkos::Array<size_t, _rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);

      temp += (this->params.gamma_id - this->params.gammas[mu]) * 0.5 *
              xp.second * (this->g_in(Idcs..., mu) * this->s_in(xp.first));
      temp += (this->params.gamma_id + this->params.gammas[mu]) * 0.5 *
              xm.second *
              (conj(this->g_in(xm.first, mu)) * this->s_in(xm.first));
    }

    this->s_out(Idcs...) += this->s_in(Idcs...) - this->params.kappa * temp;
  }

  // only for testing purpose, not the real Ddagger operator
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::TagDdagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
#pragma unroll
    for (size_t mu = 0; mu < _rank; ++mu) {
      auto xm = shift_index_minus_bc<_rank, size_t>(
          Kokkos::Array<size_t, _rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<_rank, size_t>(
          Kokkos::Array<size_t, _rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);

      temp += (this->params.gamma_id - this->params.gammas[mu]) * 0.5 *
              xp.second * (this->g_in(Idcs..., mu) * this->s_in(xp.first));
      temp += (this->params.gamma_id + this->params.gammas[mu]) * 0.5 *
              xm.second *
              (conj(this->g_in(xm.first, mu)) * this->s_in(xm.first));
    }

    this->s_out(Idcs...) += this->s_in(Idcs...) - this->params.kappa * temp;
  }
};
// Deduction guide
template <typename GaugeType, typename ParamType>
WilsonDiracOperator(const GaugeType&, const ParamType&)
    -> WilsonDiracOperator<ParamType::rank, ParamType::Nc, ParamType::RepDim>;

template <size_t _rank, size_t _Nc, size_t _RepDim>
class HWilsonDiracOperator
    : public DiracOperator<HWilsonDiracOperator<_rank, _Nc, _RepDim>, _rank,
                           _Nc, _RepDim> {
 public:
  constexpr static size_t Nc = _Nc;
  constexpr static size_t RepDim = _RepDim;
  constexpr static size_t rank = _rank;

  ~HWilsonDiracOperator() = default;
  using Base =
      DiracOperator<HWilsonDiracOperator<rank, Nc, RepDim>, rank, Nc, RepDim>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
#pragma unroll
    for (size_t mu = 0; mu < _rank; ++mu) {
      auto xm = shift_index_minus_bc<_rank, size_t>(
          Kokkos::Array<size_t, _rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);
      auto xp = shift_index_plus_bc<_rank, size_t>(
          Kokkos::Array<size_t, _rank>{Idcs...}, mu, 1, 3, -1,
          this->params.dimensions);

      temp += (this->params.gamma_id - this->params.gammas[mu]) * 0.5 *
              xp.second * (this->g_in(Idcs..., mu) * this->s_in(xp.first));
      temp += (this->params.gamma_id + this->params.gammas[mu]) * 0.5 *
              xm.second *
              (conj(this->g_in(xm.first, mu)) * this->s_in(xm.first));
    }

    this->s_out(Idcs...) +=
        this->params.gamma5 * (this->s_in(Idcs...) - this->params.kappa * temp);
  }

  // only for testing porpose, not the real Ddagger operator
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::TagDdagger,
                                              const Indices... Idcs) const {
    operator()(typename Base::TagD(), Idcs...);
  }
};
// Deduction guide
template <typename GaugeType, typename ParamType>
HWilsonDiracOperator(const GaugeType&, const ParamType&)
    -> HWilsonDiracOperator<ParamType::rank, ParamType::Nc, ParamType::RepDim>;

}  // namespace klft