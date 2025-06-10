#pragma once
#include "GLOBAL.hpp"

namespace klft {

template <size_t _rank, size_t _Nc> struct HMC_Params {
  static constexpr const size_t rank = _rank;
  static constexpr const size_t Nc = _Nc;
};
} // namespace klft
