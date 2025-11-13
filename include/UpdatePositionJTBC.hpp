#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "UpdatePosition.hpp"

namespace klft {
template <size_t rank, size_t Nc>
class UpdatePositionGaugeJTBC : public UpdatePosition {
 public:
  using GaugeFieldType =
      typename DeviceGaugeFieldType<rank, Nc, GaugeFieldKind::JTBC>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  GaugeFieldType gauge_field;
  AdjFieldType adjoint_field;
  real_t eps;
  size_t step{0};
  const size_t total_steps;

  UpdatePositionGaugeJTBC() = delete;
  ~UpdatePositionGaugeJTBC() = default;

  UpdatePositionGaugeJTBC(const GaugeFieldType& gauge_field_,
                          AdjFieldType& adjoint_field_,
                          size_t steps_)
      : UpdatePosition(0),
        gauge_field(gauge_field_),
        adjoint_field(adjoint_field_),
        total_steps(steps_),
        eps(0.0) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      gauge_field.field(Idcs..., mu) =
          expoSUN(eps * adjoint_field(Idcs..., mu)) *
          gauge_field.field(Idcs..., mu);
    }
  }

  real_t defect_value(real_t t) const {
    // 0<= t <=1
    // returns a defect value for the given t
    real_t k = 50;
    real_t ret = (1.0 - Kokkos::exp(-k * t)) / (1.0 - Kokkos::exp(-k));
    ret = ret <= 0.0 ? 0.0 : ret;
    ret = ret > 1.0 ? 1.0 : ret;
    // ret = 1.0;
    return ret;
  }

  void mod_defect() {
    // this interpolating the defect from <1 to 1 along the trajectory
    real_t t = real_t(step) / real_t(total_steps - 1);
    real_t next_defect = defect_value(t);
    if (KLFT_VERBOSITY > 3) {
      Kokkos::printf("t=%f, defect set to %f\n", t, next_defect);
    }
    gauge_field.template set_defect<index_t>(next_defect);
  }

  void update(const real_t step_size) override {
    eps = step_size;
    IndexArray<rank> start;
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }

    tune_and_launch_for<rank>("UpdatePositionGauge", start,
                              gauge_field.dimensions, *this);
    Kokkos::fence();
    mod_defect();
    ++step;
    if (step >= total_steps) {
      step = 0;
    }
  }
};
}  // namespace klft
