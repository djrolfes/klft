#pragma once
#include "GLOBAL.hpp"
#include "Spinor.hpp"
#include "Tuner.hpp"
namespace klft {
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorPointSource {
  static const size_t rank = 4;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorPointSource() = default;
  deviceSpinorPointSource(const index_t L0, const index_t L1, const index_t L2,
                          const index_t L3,
                          const IndexArray<RepDim>& sourcePoint,
                          const index_t& deltaIndex)
      : dimensions({L0, L1, L2, L3}),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(L0, L1, L2, L3, field, sourcePoint, deltaIndex);
  };
  deviceSpinorPointSource(const IndexArray<4>& dimensions,
                          const IndexArray<RepDim>& sourcePoint,
                          const index_t& deltaIndex)
      : dimensions(dimensions),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            sourcePoint, deltaIndex);
  };

  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, SpinorField<Nc, RepDim>& V,
               const IndexArray<RepDim>& sourcePoint,
               const index_t& deltaIndex) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<RepDim>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          IndexArray<RepDim>{i0, i1, i2, i3} == sourcePoint
              ? V(i0, i1, i2, i3) = deltaSpinor<Nc, RepDim>()
              : V(i0, i1, i2, i3) = zeroSpinor<Nc, RepDim>();
        });
    Kokkos::fence();
  }
  SpinorField<Nc, RepDim> field;
  IndexArray<RepDim> dimensions;
  IndexArray<RepDim> sourcePoint;
  index_t deltaIndex;
};

}  // namespace klft
