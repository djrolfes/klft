#pragma once
#include "AdjointSUN.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"

namespace klft {
struct WilsonFlowParams {
  // define parameters for a given Wilson Flow calculation, the wilson flow
  // follows arxiv.org/pdf/1006.4518
  // flow time
  real_t tau;
  // number of Wilson flow steps that should be generated
  index_t n_steps;
  // flow time step size
  real_t eps;
  // beta TODO: do not use the beta here
  real_t beta;

  WilsonFlowParams() {
    // default parameters (eps = 0.01)
    n_steps = 10;
    tau = 1.0;
    eps = real_t(tau / n_steps);
    beta = real_t(1.0);
  }
};

template <typename DGaugeFieldType> struct WilsonFlow {
  // implement the Wilson flow, for now the field will not be copied, but it
  // will be flown in place -> copying needs to be done before
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  static_assert(rank == 4); // The wilson flow is only defined for 4D Fields
  WilsonFlowParams params;

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  GaugeFieldT field;
  GaugeFieldT tmp_staple;
  SUNAdjField<rank, Nc> tmp_Z;
  index_t current_step{0};

  WilsonFlow() = delete;

  WilsonFlow(const GaugeFieldT &_field, WilsonFlowParams &_params)
      : params(_params), field(_field.field), tmp_staple(_field.field) {
    const IndexArray<rank> dims = _field.dimensions;
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z, dims[0], dims[1],
                    dims[2], dims[3]);
    // Kokkos::realloc(Kokkos::WithoutInitializing, tmp_staple, dims[0],
    // dims[1],
    //                 dims[2], dims[3]);

    //            Kokkos::deep_copy(_field.field, tmp_stap);
    //            Kokkos::deep_copy(_field.field, tmp_Z);
    Kokkos::fence();
  }

  // execute the wilson flow
  void flow() { // todo: check this once by saving a staple field and once by
                // locally calculating the staple
    for (int step = 0; step < params.n_steps; ++step) {

#pragma unroll
      for (index_t fstep = 0; fstep < 3; ++fstep) {
        this->current_step = fstep;
        stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
        Kokkos::fence();

        tune_and_launch_for<rank>("Wilsonflow-flow",
                                  IndexArray<rank>{0, 0, 0, 0},
                                  field.dimensions, *this);
        Kokkos::fence();
      }
    }
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW1(indexType i0, indexType i1, indexType i2,
                                     indexType i3, index_t mu) const {
    SUN<Nc> Z0_SUN = field.field(i0, i1, i2, i3, mu) *
                     (tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z0 = traceT(Z0_SUN) * (-1.0 * params.eps);
    tmp_Z(i0, i1, i2, i3, mu) = Z0; // does this need to be deep copied?
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z0 * static_cast<real_t>(1.0 / 4.0)) *
        field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW2(indexType i0, indexType i1, indexType i2,
                                     indexType i3, index_t mu) const {
    SUN<Nc> Z1_SUN = field.field(i0, i1, i2, i3, mu) *
                     (tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z1 = traceT(Z1_SUN) * (-1.0 * params.eps);
    SUNAdj<Nc> Z0 = tmp_Z(i0, i1, i2, i3, mu);
    Z1 = Z1 * static_cast<real_t>(8.0 / 9.0) -
         Z0 * static_cast<real_t>(17.0 / 36.0);
    tmp_Z(i0, i1, i2, i3, mu) = Z1;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z1) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepV(indexType i0, indexType i1, indexType i2,
                                    indexType i3, index_t mu) const {
    SUN<Nc> Z2_SUN = tmp_staple.field(i0, i1, i2, i3, mu) *
                     (field.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z2 = traceT(Z2_SUN) * (-1.0 * params.eps);
    SUNAdj<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
    Z2 = (Z2 * static_cast<real_t>(3.0 / 2.0) - Z_old);
    // SUNAdj<Nc> tmp = (Z2);
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z2) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(const indexType i0, const indexType i1,
                                         const indexType i2,
                                         const indexType i3) const {
#pragma unroll
    for (index_t mu = 0; mu < 4; ++mu) {
      switch (this->current_step) {
      case 0:
        stepW1(i0, i1, i2, i3, mu);
        break;
      case 1:
        stepW2(i0, i1, i2, i3, mu);
        break;
      case 2:
        stepV(i0, i1, i2, i3, mu);
        break;
      }
    }
  }
};

} // namespace klft
