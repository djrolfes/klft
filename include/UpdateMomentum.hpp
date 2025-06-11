
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
      DeviceGaugeFieldTypeTraits<GaugeFieldType>::rank;
  static constexpr size_t Nc = DeviceGaugeFieldTypeTraits<GaugeFieldType>::Nc;
  static constexpr GaugeFieldType kind =
      DeviceGaugeFieldTypeTraits<GaugeFieldType>::kind;

  const IndexArray<rank> dimensions;
  real_t eps;

  UpdateMomentumGauge() = delete;
  ~UpdateMomentumGauge() = default;

  // TODO: Add Force application
};

} // namespace klft
