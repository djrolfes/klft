#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugePlaquette.hpp"
#include "Kokkos_Atomic.hpp"

namespace klft {
template <typename DGaugeFieldType>
void get_sp_distribution(const typename DGaugeFieldType::type gauge_field,
                         std::vector<real_t> &rtn, real_t max = 0.1,
                         real_t bin_width = 0.001) {
  static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4);
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));

  // number of bins
  size_t nbins = static_cast<size_t>(max / bin_width);
  rtn.resize(nbins, 0.0);

  // 1. Create a Kokkos View on the device for the histogram.
  // Initialize it to zeros.
  Kokkos::View<real_t *> rtn_d("rtn_d", nbins);
  Kokkos::deep_copy(rtn_d, 0.0);

  GaugePlaq<Nd, Nc, k> GPlaq(gauge_field, plaq_per_site,
                             gauge_field.dimensions);

  tune_and_launch_for<Nd>("compute plaq_per_site", IndexArray<Nd>{0},
                          gauge_field.dimensions, GPlaq);
  Kokkos::fence();

  tune_and_launch_for<Nd>(
      "binning sp's", IndexArray<Nd>{0}, gauge_field.dimensions,
      KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3) {
        real_t s = Kokkos::real(2.0 - plaq_per_site(i0, i1, i2, i3));
        // now find the end bin:
        index_t endbin =
            static_cast<index_t>(nbins * Kokkos::min(1.0, s / max));
        for (index_t i = 0; i < endbin; i++) {
          Kokkos::atomic_inc(&rtn_d[i]);
        }
      });
  Kokkos::fence();

  Kokkos::View<real_t *>::HostMirror rtn_h = Kokkos::create_mirror_view(rtn_d);

  // Perform the device-to-host copy.
  Kokkos::deep_copy(rtn_h, rtn_d);

  // 4. Copy data from the host mirror into the final std::vector.
  for (size_t i = 0; i < nbins; ++i) {
    rtn[i] = rtn_h(i);
  }

  auto volume = 1;
  for (size_t vol : gauge_field.dimensions) {
    volume *= vol;
  }
  for (index_t i = 0; i < size(rtn); i++) {
    rtn[i] /= volume;
  }
}

template <typename DGaugeFieldType>
real_t get_spmax(const typename DGaugeFieldType::type gauge_field) {
  static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4);
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));

  // number of bins
  GaugePlaq<Nd, Nc, k> GPlaq(gauge_field, plaq_per_site,
                             gauge_field.dimensions);

  tune_and_launch_for<Nd>("compute plaq_per_site", IndexArray<Nd>{0},
                          gauge_field.dimensions, GPlaq);

  Kokkos::fence();
  real_t rtn = 0.0;
  auto policy = Policy<Nd>({0, 0, 0, 0}, gauge_field.dimensions);
  Kokkos::parallel_reduce(
      "get h (sp_max)", policy,
      KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3,
                    real_t & local_max) {
        // GPlaq(i0, i1, i2, i3);
        real_t s = Kokkos::real(2.0 - plaq_per_site(i0, i1, i2, i3));
        local_max = Kokkos::max(local_max, s);
      },
      Kokkos::Max<real_t>(rtn));
  Kokkos::fence();

  return rtn;
}

} // namespace klft
