#pragma once
#include "ActionDensity.hpp"
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
  bool dynamical_flow;
  real_t min_flow_time;
  real_t max_flow_time;
  real_t sp_max_target;
  real_t t_sqrd_E_target;
  size_t first_tE_measure_step;
  bool log_details;
  std::string wilson_flow_filename;

  std::vector<std::string> log_strings;

  WilsonFlowParams() {
    // default parameters (eps = 0.01)
    n_steps = 10;
    tau = 1.0;
    eps = real_t(tau / n_steps);
    dynamical_flow = false;
    min_flow_time = -1.0;
    max_flow_time = -1.0;
    sp_max_target = real_t(0.067);
    t_sqrd_E_target = real_t(0.1); // this is Nc dependent
    first_tE_measure_step = 10;
    log_details =
        false; // for now this will only do something for the dynamical wflow
    wilson_flow_filename = "";
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
  WilsonFlowParams &params;

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  GaugeFieldT field;
  GaugeFieldT tmp_staple;
  SUNAdjField<rank, Nc> tmp_Z;
  index_t current_step{0};
  real_t eps{0.01};

  WilsonFlow() = delete;

  WilsonFlow(const GaugeFieldT &_field, WilsonFlowParams &_params)
      : params(_params), field(_field.field), tmp_staple(_field.field) {
    const IndexArray<rank> dims = _field.dimensions;
    eps = params.eps;
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z, dims[0], dims[1],
                    dims[2], dims[3]);
    Kokkos::fence();
  }

  void flow_step() {

#pragma unroll
    for (index_t fstep = 0; fstep < 3; ++fstep) {
      this->current_step = fstep;
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      tune_and_launch_for<rank>("Wilsonflow-flow", IndexArray<rank>{0, 0, 0, 0},
                                field.dimensions, *this);
      Kokkos::fence();
    }
  }

  // execute the wilson flow
  void flow() { // todo: check this once by saving a staple field and once by
                // locally calculating the staple
    if (params.dynamical_flow) {
      if (KLFT_VERBOSITY > 2) {
        printf("Using dynamical Wilson flow...\n");
      }
      flow_dynamical();
      return;
    }
    for (int step = 0; step < params.n_steps; ++step) {
      flow_step();
    }
  }

  void flow_dynamical() {
    // dynamically does the wilson flow either until sp_max is below
    // sp_max_target or until t^2E is above t_sqd_E_target.
    bool continue_flow = true;
    real_t sp_max;
    size_t step_t{0};
    size_t measure_step{params.first_tE_measure_step};
    size_t measure_step_old{0};
    real_t t_sqd_E{0};
    real_t t_sqd_E_old{0};
    while (continue_flow) {
      flow_step();
      step_t++;
      this->params.tau = step_t * params.eps;

      if (this->params.tau > this->params.min_flow_time) {

        // quick napkin math:
        // each flow_step() call calculates the staple field 3 times and
        // multiplies this to the field(...) each time. A staple_field * field
        // should be a bit less than 2 plaquette calculations in terms of
        // instruction count. The get_spmax call does one sweep over the GPlaq
        // Functor within a parallel reduction.
        // -> a get_spmax call (- the overhead from the reduction vs. the
        // parallel for) should be about(slightly larger than) 1/6'th of a
        // flow_step() call
        sp_max = get_spmax<DGaugeFieldType>(this->field);

        if (step_t >= measure_step) {
          // linearlise the measured t_sqd_E to get a prediction for the end
          // step
          measure_step = step_t;
          t_sqd_E = getActionDensity_clover<DGaugeFieldType>(this->field) *
                    params.tau * (params.tau);
          real_t slope = (t_sqd_E - t_sqd_E_old) / (step_t - measure_step_old);
          real_t intercept = t_sqd_E - slope * static_cast<real_t>(step_t);
          measure_step_old = measure_step;
          measure_step = static_cast<index_t>(
              ((params.t_sqrd_E_target - intercept) / slope) + 1);
          if (KLFT_VERBOSITY > 1) {
            printf("Wilson Flow prediction: next measurement at step %zu\n",
                   measure_step);
            printf("  slope: %1.6f, intercept: %1.6f\n", slope, intercept);
            printf("  current t^2E: %1.6f\n", t_sqd_E);
          }
        }
        if (KLFT_VERBOSITY > 4) {
          printf("Wilson Flow step %zu: sp_max = %1.6f, t^2E = %1.6f\n", step_t,
                 sp_max, t_sqd_E);
        }
        if ((sp_max <= params.sp_max_target ||
             t_sqd_E >= params.t_sqrd_E_target)) {
          continue_flow = false;
        }
      }

      if (step_t * params.eps >= params.max_flow_time &&
          params.max_flow_time > 0.0) {
        continue_flow = false;
      }
    }
    if (KLFT_VERBOSITY > 1) {
      printf("Wilson Flow completed in %zu steps, total flow time %1.6f\n",
             step_t, step_t * params.eps);
    }
    if (params.log_details) {
      log_flow_step_dyn(step_t, params.tau, sp_max, t_sqd_E, measure_step,
                        getActionDensity_clover<DGaugeFieldType>(this->field) *
                            params.tau * params.tau);
    }
  }

  void log_flow_step_dyn(size_t steps, real_t tau, real_t sp_max,
                         real_t t_sqrd_E_old, size_t measure_step,
                         real_t t_sqrd_E) {
    if (!params.log_details) {
      return;
    }
    std::string log_line =
        std::to_string(steps) + ", " + std::to_string(tau) + ", " +
        std::to_string(sp_max) + ", " + std::to_string(t_sqrd_E_old) + ", " +
        std::to_string(measure_step) + ", " + std::to_string(t_sqrd_E);
    params.log_strings.push_back(log_line);
  }

  void flow_DBW2() { // todo: check this once by saving a staple field and
                     // once by locally calculating the staple
    for (int step = 0; step < params.n_steps; ++step) {

#pragma unroll
      for (index_t fstep = 0; fstep < 3; ++fstep) {
        this->current_step = fstep;
        stapleField<DGaugeFieldType>(this->field, this->tmp_staple, -1.4088);
        Kokkos::fence();

        tune_and_launch_for<rank>("Wilsonflow-flow-DBW2",
                                  IndexArray<rank>{0, 0, 0, 0},
                                  field.dimensions, *this);
        Kokkos::fence();
      }
    }
  }

  void flow_impr(real_t b1) {
    for (int step = 0; step < params.n_steps; ++step) {

#pragma unroll
      for (index_t fstep = 0; fstep < 3; ++fstep) {
        this->current_step = fstep;
        stapleField<DGaugeFieldType>(this->field, this->tmp_staple, -1.4088);
        Kokkos::fence();

        tune_and_launch_for<rank>("Wilsonflow-flow-impr",
                                  IndexArray<rank>{0, 0, 0, 0},
                                  field.dimensions, *this);
        Kokkos::fence();
      }
    }
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW1(indexType i0, indexType i1, indexType i2,
                                     indexType i3, index_t mu) const {
    // SUN<Nc> Z0_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z0_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z0 = 2.0 * traceT(Z0_SUN); //* (Nc / params.beta);
    tmp_Z(i0, i1, i2, i3, mu) = Z0;       // does this need to be deep copied?
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z0 * -0.25 * this->eps) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW2(indexType i0, indexType i1, indexType i2,
                                     indexType i3, index_t mu) const {
    // SUN<Nc> Z1_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z1_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z1 = 2.0 * traceT(Z1_SUN); // * (Nc / params.beta);
    SUNAdj<Nc> Z0 = tmp_Z(i0, i1, i2, i3, mu);
    Z1 = Z1 * static_cast<real_t>(8.0 / 9.0) -
         Z0 * static_cast<real_t>(17.0 / 36.0);
    tmp_Z(i0, i1, i2, i3, mu) = Z1;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z1 * -this->eps) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepV(indexType i0, indexType i1, indexType i2,
                                    indexType i3, index_t mu) const {
    // SUN<Nc> Z2_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z2_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z2 = 2.0 * traceT(Z2_SUN); // * (Nc / params.beta);
    SUNAdj<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
    Z2 = (Z2 * 0.75 - Z_old);
    // SUNAdj<Nc> tmp = (Z2);
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z2 * -this->eps) * field.field(i0, i1, i2, i3, mu);
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
