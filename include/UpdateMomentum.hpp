
#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"

namespace klft {

class UpdateMomentum : public std::enable_shared_from_this<UpdateMomentum> {
public:
  UpdateMomentum() = delete;
  virtual ~UpdateMomentum() = default;

  // update the momentum of the system
  // using the given step size
  virtual void update(const real_t step_size) = 0;
};

template <typename DGaugeFieldType,
          typename DAdjFieldType> // TODO: Add a check, that DGaugeFieldType
                                  // actually is a DeviceGaugeFieldType
class UpdateMomentumGauge : public UpdateMomentum {
public:
  using GaugeFieldType = typename DGaugeFieldType::type;
  using AdjFieldType = typename DAdjFieldType::type;
  static constexpr size_t rank =
      DeviceGaugeFieldTypeTraits<GaugeFieldType>::Rank;
  static constexpr size_t Nc = DeviceGaugeFieldTypeTraits<GaugeFieldType>::Nc;
  static constexpr GaugeFieldType kind =
      DeviceGaugeFieldTypeTraits<GaugeFieldType>::Kind;
  GaugeFieldType gauge_field;
  AdjFieldType adjoint_field;
  const real_t beta;

  const IndexArray<rank> dimensions;
  real_t eps;
  constGaugeField<rank, Nc> staple_field;

  UpdateMomentumGauge() = delete;
  ~UpdateMomentumGauge() = default;

  UpdateMomentumGauge(const GaugeFieldType gauge_field_,
                      const AdjFieldType adjoint_field_,
                      const IndexArray<rank> dimensions_, const real_t beta_)
      : gauge_field(gauge_field_), adjoint_field(adjoint_field_),
        dimensions(dimensions_), beta(beta_), eps(0.0) {}
  // todo: Add Force as a function instead of it being incorporated into the
  // functor.

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // Update the momentum field
    adjoint_field(Idcs...) -=
        this->eps * (this->beta / this->Nc) *
        traceT(this->gauge_field(Idcs...) * this->staple_field(Idcs...));
  }

  void update(const real_t step_size) override {
    eps = step_size;
    IndexArray<rank> start;
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    // launch the kernels
    staple_field = stapleField(gauge_field.field);
    tune_and_launch_for<rank>("UpdateMomentumGauge", start, dimensions, *this);
    Kokkos::fence();
  }
};

} // namespace klft
