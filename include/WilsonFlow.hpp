#pragma once
#include <string>
#include "ActionDensity.hpp"
#include "AdjointSUN.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"
#include "Kokkos_Complex.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "Kokkos_Printf.hpp"
#include "decl/Kokkos_Declare_OPENMP.hpp"
#include "impl/Kokkos_HostThreadTeam.hpp"

namespace klft {

enum class WilsonFlowStyle { RK3, RK4, Adaptive, Dynamic };

struct WilsonFlowParams_Dynamic {
  real_t min_flow_time;
  real_t max_flow_time;
  real_t sp_max_target;
  real_t t_sqrd_E_target;
  size_t first_tE_measure_step;
  bool log_details;
  std::string wilson_flow_filename;

  std::vector<std::string> log_strings;

  WilsonFlowParams_Dynamic() {
    min_flow_time = -1.0;
    max_flow_time = -1.0;
    sp_max_target = real_t(0.067);
    t_sqrd_E_target = real_t(0.1);  // this is Nc dependent
    first_tE_measure_step = 10;
    log_details =
        false;  // for now this will only do something for the dynamical wflow
    wilson_flow_filename = "";
  }
};

struct WilsonFlowParams_Adaptive {
  real_t rho;
  real_t abs_tol;
  real_t rel_tol;
  real_t max_increase;
  real_t max_decrease;

  WilsonFlowParams_Adaptive() {
    rho = 0.95;
    abs_tol = 1e-3;
    rel_tol = 1e-1;
    max_increase = 1.1;
    max_decrease = 0.6;
  }
};

struct WilsonFlowParams {
  // define parameters for a given Wilson Flow calculation, the wilson flow
  // follows arxiv.org/pdf/1006.4518
  // flow time
  real_t tau;
  // number of Wilson flow steps that should be generated
  index_t n_steps;
  // flow time step size
  real_t eps;
  WilsonFlowStyle style;
  WilsonFlowParams_Adaptive adaptiveParams;
  WilsonFlowParams_Dynamic dynamicParams;

  WilsonFlowParams() {
    // default parameters (eps = 0.01)
    n_steps = 10;
    tau = 1.0;
    eps = real_t(tau / n_steps);
    style = WilsonFlowStyle::RK3;
  }
};

struct WilsonFlowData {
  size_t step;
  real_t flow_time;
  real_t sp_max_init;
  real_t sp_max;
  real_t sp_max_old;
  real_t sp_max_deriv;
  real_t t_sqrd_E;
  real_t t_sqrd_E_old;
  size_t measure_step;
  size_t measure_step_old;

  std::string to_string() {
    return std::to_string(step) + ", " + std::to_string(flow_time) + ", " +
           std::to_string(sp_max_init) + ", " + std::to_string(sp_max) + ", " +
           std::to_string(sp_max_deriv) + ", " + std::to_string(t_sqrd_E_old) +
           ", " + std::to_string(measure_step) + ", " +
           std::to_string(t_sqrd_E);
  }

  std::string header_string() {
    return "# step, flow_step, flow_time, sp_max_init, sp_max, sp_max_deriv, "
           "tsquaredxaction_density(old), next_measure_t^E_step, "
           "tsquaredxaction_density\n";
  }
};

template <typename DGaugeFieldType>
struct WilsonFlow {
  // implement the Wilson flow, for now the field will not be copied, but it
  // will be flown in place -> copying needs to be done before
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  static_assert(rank == 4);  // The wilson flow is only defined for 4D Fields
  WilsonFlowParams& params;
  WilsonFlowData wfdata;

  struct RK3 {};
  struct RK4 {};

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  GaugeFieldT field;
  GaugeFieldT field_bak;
  GaugeFieldT tmp_staple;
  SUNAdjField<rank, Nc> tmp_Z;
  SUNAdjField<rank, Nc> tmp_Z_err;
  index_t current_step{0};
  real_t eps{0.01};

  WilsonFlow() = delete;

  WilsonFlow(const GaugeFieldT& _field, WilsonFlowParams& _params)
      : params(_params),
        field(_field.field),
        field_bak(_field.field),
        tmp_staple(_field.field),
        wfdata() {
    const IndexArray<rank> dims = _field.dimensions;
    eps = params.eps;
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z, dims[0], dims[1],
                    dims[2], dims[3]);
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z_err, dims[0], dims[1],
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

  void flow_step_RK4() {
#pragma unroll
    for (index_t fstep = 0; fstep < 6; ++fstep) {
      this->current_step = fstep;
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      tune_and_launch_for<rank, RK4>("Wilsonflow-flow",
                                     IndexArray<rank>{0, 0, 0, 0},
                                     field.dimensions, *this);
      Kokkos::fence();
    }
  }

  // execute the wilson flow
  void flow() {  // todo: check this once by saving a staple field and once by
                 // locally calculating the staple
    switch (params.style) {
      case WilsonFlowStyle::RK3:
        if (KLFT_VERBOSITY > 2) {
          printf("Using RK3 Wilson flow...\n");
        }
        for (int step = 0; step < params.n_steps; ++step) {
          flow_step();
        }
        break;
      case WilsonFlowStyle::RK4:
        if (KLFT_VERBOSITY > 2) {
          printf("Using RK4 Wilson flow...\n");
        }
        for (int step = 0; step < params.n_steps; ++step) {
          flow_step_RK4();
        }
        break;
      case WilsonFlowStyle::Adaptive:
        if (KLFT_VERBOSITY > 2) {
          printf("Using adaptive Wilson flow...\n");
        }
        flow_adaptive();
        break;
      case WilsonFlowStyle::Dynamic:
        if (KLFT_VERBOSITY > 2) {
          printf("Using dynamical Wilson flow...\n");
        }
        flow_dynamical();
        break;
    }
  }

  void flow_adaptive() {
    real_t flow_time = 0.0;

    // Allocate temporary Views for the Lie Algebra stages (K1, K2, K3)
    // We can reuse tmp_Z for Z calculations and tmp_Z_err for one of the K's if
    // needed, but explicit allocation is clearer and safer for the
    // accumulators.
    SUNAdjField<rank, Nc> K1("K1", field.dimensions[0], field.dimensions[1],
                             field.dimensions[2], field.dimensions[3]);
    SUNAdjField<rank, Nc> K2("K2", field.dimensions[0], field.dimensions[1],
                             field.dimensions[2], field.dimensions[3]);
    SUNAdjField<rank, Nc> K3("K3", field.dimensions[0], field.dimensions[1],
                             field.dimensions[2], field.dimensions[3]);

    if (KLFT_VERBOSITY > 2) {
      Kokkos::printf("Starting faithful RK-MK (2)3 adaptive Wilson flow\n");
    }

    // Parameters from paper/struct
    auto aparams = params.adaptiveParams;

    // Loop until target flow time is reached
    while (flow_time < params.tau) {
      // 0. Backup current field (V0)
      Kokkos::deep_copy(field_bak.field, field.field);

      // --- Stage 1 ---
      // Y1 = 0, so V = V0.
      // Calculate Z(V0).
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      // K1 = f_0(0, Z) + B_1[0, Z] = Z. (Since Y1=0)
      auto Z = this->tmp_Z;  // Re-use tmp_Z member as a handle for Z
      calc_Z_LieAlgebra(this->field, this->tmp_staple,
                        Z);  // Helper to compute Z from staple*U
      Kokkos::deep_copy(K1, Z);

      // --- Stage 2 ---
      // Y2 = h * (1/2) * K1 [cite: 137]
      // V2 = exp(Y2) * V0
      update_field_RKMK(field_bak, field, K1, 0.5 * eps);
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      calc_Z_LieAlgebra(this->field, this->tmp_staple, Z);
      // K2 = Z - 0.5 * [Y2, Z]
      // Note: We need to reconstruct Y2 inside the kernel or pass the
      // coefficient.
      compute_stage_K(K2, Z, K1, 0.5 * eps);

      // --- Stage 3 ---
      // Y3 = h * (3/4) * K2 [cite: 138]
      // V3 = exp(Y3) * V0
      update_field_RKMK(field_bak, field, K2, 0.75 * eps);
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      calc_Z_LieAlgebra(this->field, this->tmp_staple, Z);
      // K3 = Z - 0.5 * [Y3, Z]
      compute_stage_K(K3, Z, K2, 0.75 * eps);

      // --- Stage 4 (Final Solution V1) ---
      // Omega_3 = h * (2/9 K1 + 1/3 K2 + 4/9 K3) [cite: 139]
      // V1 = exp(Omega_3) * V0
      // We perform the weighted sum and exp map in one go.
      update_field_RKMK_final(field_bak, field, K1, K2, K3, eps);
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      // Calculate Z(V1) for the error estimate (FSAL property / Embedded
      // method) This corresponds to \hat{K}_4 in the paper (order 2 approx).
      // [cite: 129]
      calc_Z_LieAlgebra(this->field, this->tmp_staple, Z);

      // --- Error Estimate ---
      // Err = || Omega_2 - Omega_3 ||
      // Omega_2 uses weights: 7/24, 1/4, 1/3, 1/8 [cite: 141]
      // Omega_3 uses weights: 2/9, 1/3, 4/9, 0
      // Diff weights: (7/24 - 2/9), (1/4 - 1/3), (1/3 - 4/9), 1/8
      // Diff = h * ( 5/72 K1 - 1/12 K2 - 1/9 K3 + 1/8 Z )

      real_t err = 0.0;
      real_t total_volume =
          this->field.dimensions[0] * this->field.dimensions[1] *
          this->field.dimensions[2] * this->field.dimensions[3] * 4.0;

      Kokkos::parallel_reduce(
          "WilsonFlow_Error_Calc",
          Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                    field.dimensions),
          KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                        const index_t i3, real_t& local_err) {
            for (index_t mu = 0; mu < 4; ++mu) {
              SUNAdj<Nc> k1_loc = K1(i0, i1, i2, i3, mu);
              SUNAdj<Nc> k2_loc = K2(i0, i1, i2, i3, mu);
              SUNAdj<Nc> k3_loc = K3(i0, i1, i2, i3, mu);
              SUNAdj<Nc> z_loc = Z(i0, i1, i2, i3, mu);

              // Calculate Omega_3 (Accumulator) for normalization if using
              // Relative Tol
              SUNAdj<Nc> omega_3 =
                  (k1_loc * (2.0 / 9.0) + k2_loc * (1.0 / 3.0) +
                   k3_loc * (4.0 / 9.0)) *
                  eps;

              // Calculate Difference
              SUNAdj<Nc> diff = (k1_loc * (5.0 / 72.0) - k2_loc * (1.0 / 12.0) -
                                 k3_loc * (1.0 / 9.0) + z_loc * (1.0 / 8.0)) *
                                eps;

              // Paper  error metric
              // Frobenius norm squared of diff
              real_t diff_norm = Kokkos::sqrt(norm2<Nc>(diff));
              real_t omega_norm = Kokkos::sqrt(norm2<Nc>(omega_3));

              real_t scale = aparams.abs_tol + aparams.rel_tol * omega_norm;
              local_err += Kokkos::pow(diff_norm / scale, 2);
            }
          },
          err);
      Kokkos::fence();

      err = Kokkos::sqrt(err / total_volume);

      // --- Step Size Control [cite: 104, 106] ---
      // Order p=2 used for prediction, so exponent is 1/(p+1) = 1/3
      real_t eps_opt =
          eps * Kokkos::pow(1.0 / (err + 1e-10), 1.0 / 3.0) * aparams.rho;
      eps_opt = Kokkos::min(aparams.max_increase * eps,
                            Kokkos::max(aparams.max_decrease * eps, eps_opt));

      if (KLFT_VERBOSITY > 2) {
        Kokkos::printf("Adaptive Step: t=%1.4f, h=%e, err=%e, new_h=%e\n",
                       flow_time, eps, err, eps_opt);
      }

      if (err <= 1.0) {
        // Accept
        flow_time += eps;
      } else {
        // Reject: Restore field
        Kokkos::deep_copy(field.field, field_bak.field);
        Kokkos::fence();
      }

      // Update step size
      eps = eps_opt;
      if (flow_time + eps > params.tau) {
        eps = params.tau - flow_time + REAL_T_EPSILON;
      }
    }
  }

  // --- Helper Kernels for RK-MK ---

  // Calculates Z (Lie Algebra force) from the Field and Staple
  // Z = P_Alg ( U * Staple )
  void calc_Z_LieAlgebra(GaugeFieldT& f,
                         GaugeFieldT& st,
                         SUNAdjField<rank, Nc>& out_Z) {
    Kokkos::parallel_for(
        "Calc_Z",
        Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                  f.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (int mu = 0; mu < 4; ++mu) {
            SUN<Nc> prod =
                f.field(i0, i1, i2, i3, mu) * st.field(i0, i1, i2, i3, mu);
            // Project to algebra (traceless anti-hermitian part)
            // Corresponds to [cite: 118] Z mapping
            out_Z(i0, i1, i2, i3, mu) = 2.0 * traceT(prod);
          }
        });
  }

  // Updates field for internal stages: V = exp( coeff * K_prev ) * V0
  void update_field_RKMK(GaugeFieldT& v0,
                         GaugeFieldT& v_target,
                         SUNAdjField<rank, Nc>& K_prev,
                         real_t coeff) {
    Kokkos::parallel_for(
        "Update_Field_RKMK",
        Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                  v0.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (int mu = 0; mu < 4; ++mu) {
            SUNAdj<Nc> Y = K_prev(i0, i1, i2, i3, mu) * coeff;
            // V = exp(Y) * V0 [cite: 95]
            v_target.field(i0, i1, i2, i3, mu) =
                expoSUN(Y * -1.0) * v0.field(i0, i1, i2, i3, mu);
          }
        });
  }

  // Calculates K for stages: K = Z - 0.5 * [Y, Z]
  // Uses field.dimensions for the iteration policy since Z is a raw View
  void compute_stage_K(SUNAdjField<rank, Nc>& K_target,
                       SUNAdjField<rank, Nc>& Z,
                       SUNAdjField<rank, Nc>& K_prev,
                       real_t coeff) {
    Kokkos::parallel_for(
        "Compute_Stage_K",
        Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                  this->field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (int mu = 0; mu < 4; ++mu) {
            SUNAdj<Nc> z_val = Z(i0, i1, i2, i3, mu);
            SUNAdj<Nc> y_val = K_prev(i0, i1, i2, i3, mu) * coeff;

            // Commutator [Y, Z] = YZ - ZY
            SUNAdj<Nc> comm = y_val * z_val - z_val * y_val;

            // K = Z + B1 * [Y, Z], B1 = -0.5
            K_target(i0, i1, i2, i3, mu) = z_val - comm * 0.5;
          }
        });
  }

  // Final update: V1 = exp( h * (b1 K1 + b2 K2 + b3 K3) ) * V0
  void update_field_RKMK_final(GaugeFieldT& v0,
                               GaugeFieldT& v_target,
                               SUNAdjField<rank, Nc>& k1,
                               SUNAdjField<rank, Nc>& k2,
                               SUNAdjField<rank, Nc>& k3,
                               real_t h) {
    Kokkos::parallel_for(
        "Update_Final",
        Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                  v0.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (int mu = 0; mu < 4; ++mu) {
            // Weights b from Table 1, row 4 [cite: 139]
            SUNAdj<Nc> Omega = k1(i0, i1, i2, i3, mu) * (2.0 / 9.0) +
                               k2(i0, i1, i2, i3, mu) * (1.0 / 3.0) +
                               k3(i0, i1, i2, i3, mu) * (4.0 / 9.0);
            Omega = Omega * h;
            v_target.field(i0, i1, i2, i3, mu) =
                expoSUN(Omega * -1.0) * v0.field(i0, i1, i2, i3, mu);
          }
        });
  }

  void flow_dynamical() {
    // dynamically does the wilson flow either until sp_max is below
    // sp_max_target or until t^2E is above t_sqd_E_target.
    bool continue_flow = true;
    auto dparams = params.dynamicParams;
    wfdata.sp_max = 0;
    wfdata.sp_max_old = 0;
    wfdata.step = 0;
    wfdata.measure_step = dparams.first_tE_measure_step;
    wfdata.measure_step_old = 0;
    wfdata.t_sqrd_E = 0;
    wfdata.t_sqrd_E_old = 0;

    wfdata.sp_max_deriv = 0;

    wfdata.sp_max_init = get_spmax<DGaugeFieldType>(this->field);

    while (continue_flow) {
      flow_step();
      wfdata.step++;
      this->params.tau = wfdata.step * params.eps;
      wfdata.flow_time = this->params.tau;

      if (this->params.tau > dparams.min_flow_time) {
        // quick napkin math:
        // each flow_step() call calculates the staple field 3 times and
        // multiplies this to the field(...) each time. A staple_field * field
        // should be a bit less than 2 plaquette calculations in terms of
        // instruction count. The get_spmax call does one sweep over the GPlaq
        // Functor within a parallel reduction.
        // -> a get_spmax call (- the overhead from the reduction vs. the
        // parallel for) should be about(slightly larger than) 1/6'th of a
        // flow_step() call
        wfdata.sp_max = get_spmax<DGaugeFieldType>(this->field);
        if (wfdata.step > 1) {
          wfdata.sp_max_deriv += (wfdata.sp_max - wfdata.sp_max_old);
        } else {
          wfdata.sp_max_deriv += (wfdata.sp_max - wfdata.sp_max_init);
        }
        wfdata.sp_max_old = wfdata.sp_max;

        if (wfdata.step >= wfdata.measure_step) {
          // linearlise the measured wfdata.t_sqrd_E to get a prediction for the
          // end step
          wfdata.measure_step = wfdata.step;
          wfdata.t_sqrd_E =
              getActionDensity_clover<DGaugeFieldType>(this->field) *
              params.tau * params.tau;
          real_t slope = (wfdata.t_sqrd_E - wfdata.t_sqrd_E_old) /
                         (wfdata.step - wfdata.measure_step_old);
          real_t intercept =
              wfdata.t_sqrd_E - slope * static_cast<real_t>(wfdata.step);
          wfdata.measure_step_old = wfdata.measure_step;
          wfdata.measure_step = static_cast<index_t>(
              ((dparams.t_sqrd_E_target - intercept) / slope) + 1);
          wfdata.measure_step =
              wfdata.measure_step > wfdata.measure_step_old + 10
                  ? wfdata.measure_step
                  : wfdata.measure_step_old + 10;
          wfdata.measure_step =
              wfdata.measure_step < wfdata.step + 100
                  ? wfdata.measure_step
                  : wfdata.measure_step_old + 100;  // avoid too large steps
          if (KLFT_VERBOSITY > 1) {
            printf("Wilson Flow prediction: next measurement at step %zu\n",
                   wfdata.measure_step);
            printf("  slope: %1.6f, intercept: %1.6f\n", slope, intercept);
            printf("  current t^2E: %1.6f\n", wfdata.t_sqrd_E);
          }
        }
        if (KLFT_VERBOSITY > 4) {
          printf("Wilson Flow step %zu: wfdata.sp_max = %1.6f, t^2E = %1.6f\n",
                 wfdata.step, wfdata.sp_max, wfdata.t_sqrd_E);
        }
        if ((wfdata.sp_max <= dparams.sp_max_target ||
             wfdata.t_sqrd_E >= dparams.t_sqrd_E_target)) {
          continue_flow = false;
        }
      }

      if (this->params.tau >= dparams.max_flow_time &&
          dparams.max_flow_time > 0.0) {
        continue_flow = false;
      }
    }
    wfdata.sp_max_deriv =
        wfdata.sp_max_deriv / static_cast<real_t>(wfdata.step - 1);
    if (KLFT_VERBOSITY > 1) {
      printf("Wilson Flow completed in %zu steps, total flow time %1.6f\n",
             wfdata.step, wfdata.step * params.eps);
    }
    if (dparams.log_details) {
      wfdata.t_sqrd_E_old = wfdata.t_sqrd_E;
      wfdata.t_sqrd_E = getActionDensity_clover<DGaugeFieldType>(this->field) *
                        params.tau * params.tau;
      dparams.log_strings.push_back(wfdata.to_string());
    }
  }  // namespace klft

  void flow_impr(real_t b1) {
    for (int step = 0; step < params.n_steps; ++step) {
#pragma unroll
      for (index_t fstep = 0; fstep < 3; ++fstep) {
        this->current_step = fstep;
        stapleField<DGaugeFieldType>(this->field, this->tmp_staple, b1);
        Kokkos::fence();

        tune_and_launch_for<rank>("Wilsonflow-flow-impr",
                                  IndexArray<rank>{0, 0, 0, 0},
                                  field.dimensions, *this);
        Kokkos::fence();
      }
    }
  }

  void flow_DBW2() { flow_impr(-1.4011); }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW1(indexType i0,
                                     indexType i1,
                                     indexType i2,
                                     indexType i3,
                                     index_t mu) const {
    // SUN<Nc> Z0_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z0_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z0 = 2.0 * traceT(Z0_SUN);  //* (Nc / params.beta);
    tmp_Z(i0, i1, i2, i3, mu) = Z0;        // does this need to be deep copied?
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z0 * -0.25 * this->eps) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW2(indexType i0,
                                     indexType i1,
                                     indexType i2,
                                     indexType i3,
                                     index_t mu) const {
    // SUN<Nc> Z1_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z1_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z1 = 2.0 * traceT(Z1_SUN);  // * (Nc / params.beta);
    SUNAdj<Nc> Z0 = tmp_Z(i0, i1, i2, i3, mu);
    Z1 = Z1 * static_cast<real_t>(8.0 / 9.0) -
         Z0 * static_cast<real_t>(17.0 / 36.0);
    tmp_Z(i0, i1, i2, i3, mu) = Z1;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z1 * -this->eps) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepV(indexType i0,
                                    indexType i1,
                                    indexType i2,
                                    indexType i3,
                                    index_t mu) const {
    SUN<Nc> Z2_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z2 = 2.0 * traceT(Z2_SUN);  // * (Nc / params.beta);
    SUNAdj<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
    Z2 = (Z2 - Z_old);
    tmp_Z_err(i0, i1, i2, i3, mu) = Z2;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z2 * -this->eps * 0.75) * field.field(i0, i1, i2, i3, mu);
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW(indexType i0,
                                    indexType i1,
                                    indexType i2,
                                    indexType i3,
                                    index_t mu,
                                    const real_t a,
                                    const real_t b,
                                    const bool first_step,
                                    const bool last_step) const {
    SUN<Nc> Z_SUN = (field.field(i0, i1, i2, i3, mu) *
                     tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z = 2.0 * traceT(Z_SUN);  // * (Nc / params.beta);
    if (!first_step) {
      SUNAdj<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
      Z_old *= a;
      Z = (Z - Z_old);
    }
    if (!last_step) {
      tmp_Z(i0, i1, i2, i3, mu) = Z;
    } else {
      tmp_Z(i0, i1, i2, i3, mu) = Z;
    }
    Z *= -b * this->eps;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z) * field.field(i0, i1, i2, i3, mu);
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(const indexType i0,
                                         const indexType i1,
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

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(RK4,
                                         const indexType i0,
                                         const indexType i1,
                                         const indexType i2,
                                         const indexType i3) const {
    // params taken from quda codebase
    real_t a[] = {0.0,
                  0.737101392796,
                  1.634740794341,
                  0.744739003780,
                  1.469897351522,
                  2.813971388035};
    real_t b[] = {0.032918605146, 0.823256998200, 0.381530948900,
                  0.200092213184, 1.718581042715, 0.27};
#pragma unroll
    for (index_t mu = 0; mu < 4; ++mu) {
      switch (this->current_step) {
        case 0:
          stepW(i0, i1, i2, i3, mu, a[0], b[0], true, false);
          break;
        case 1:
          stepW(i0, i1, i2, i3, mu, a[1], b[1], false, false);
          break;
        case 2:
          stepW(i0, i1, i2, i3, mu, a[2], b[2], false, false);
          break;
        case 3:
          stepW(i0, i1, i2, i3, mu, a[3], b[3], false, false);
          break;
        case 4:
          stepW(i0, i1, i2, i3, mu, a[4], b[4], false, false);
          break;
        case 5:
          stepW(i0, i1, i2, i3, mu, a[5], b[5], false, true);
          break;
      }
    }
  }
};

}  // namespace klft
