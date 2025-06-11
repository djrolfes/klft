
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

  const IndexArray<rank> dimensions;
  real_t eps;
  constGaugeField<rank, Nc> staple_field;

  UpdateMomentumGauge() = delete;
  ~UpdateMomentumGauge() = default;

  // TODO: Add Force application

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // Update the momentum field
    adjoint_field(Idcs...) +=
        eps * gauge_field(Idcs...) *
        staple_field(Idcs...); // TODO: add beta as an input somewhere
  }

  void update(const real_t step_size) override {
    eps = step_size;
    IndexArray<rank> start;
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    staple_field = stapleField(gauge_field.field);
    // launch the kernel
    tune_and_launch_for<rank>("UpdateMomentumGauge", start, dimensions, *this);
    Kokkos::fence();
  }
};

} // namespace klft
