// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges.hpp>
#include <dr/detail/segments_tools.hpp>
#include <dr/mhp/views/segmented.hpp>

namespace dr::mhp {
namespace views {
namespace __detail {

struct sliding_fn {
  // maybe also add rng::range_difference_t<Rng> left_side_len,
  // rng::range_difference_t<Rng> right_side_len
  template <typename Rng>
    requires rng::viewable_range<Rng> && rng::forward_range<Rng>
  auto operator()(Rng &&r) const {
    auto halo_bounds = (&(*rng::begin(r))).halo_bounds();
    return rng::views::sliding(static_cast<Rng &&>(r),
                               halo_bounds.prev + 1 + halo_bounds.next);
  }
};

} // namespace __detail

inline constexpr __detail::sliding_fn sliding{};

} // namespace views
} // namespace dr::mhp

namespace DR_RANGES_NAMESPACE {

template <rng::range V>
  requires(dr::is_sliding_view_v<V>)
auto segments_(V &&v) {

  auto first = rng::begin(v.base());
  const auto halo_bounds = (&(*first)).halo_bounds();
  const auto sliding_view_size = rng::size(v);
  const auto orig_range_size = rng::size(v.base());
  assert(halo_bounds.prev + sliding_view_size + halo_bounds.next ==
         orig_range_size);
  assert(
      !halo_bounds.periodic); // not yet implemented, code assumes non-periodic

  return dr::mhp::views::segmented(
      v,
      dr::__detail::take_segments(
          dr::ranges::segments(first + halo_bounds.prev), sliding_view_size));
}

} // namespace DR_RANGES_NAMESPACE
