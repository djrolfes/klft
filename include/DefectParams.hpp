#pragma once
#include "GLOBAL.hpp"
namespace klft {
template <size_t Nd>
struct defectParams {
  // A struct that is used to hold the defect information for a given
  // devicePTBCGaugeField
  index_t defect_length{1};
  real_t defect_value{1.0};
  IndexArray<Nd - 1> defect_position{
      0};  // origin of the defect in mu = 1,2,3 directions
};
}  // namespace klft
