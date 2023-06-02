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

  // one can not use local algorithms if n is not equal to halo_bounds.prev + 1
  // + halo_bounds.next
  template <typename Rng, typename Int>
    requires rng::viewable_range<Rng> && rng::forward_range<Rng> &&
             rng::detail::integer_like_<Int>
  auto operator()(Rng &&r, Int n) const {
    return rng::views::sliding(static_cast<Rng &&>(r), n);
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
  const auto sliding_view_size = rng::size(v);
  // TODO: this code assumes that halo is symetric (sliding_view_size/2)
  assert(sliding_view_size % 2 == 1);
  auto first = rng::begin(v.base());
  rng::advance(first, sliding_view_size / 2);

  return dr::mhp::views::segmented(
      v, dr::__detail::take_segments(dr::ranges::segments(first),
                                     sliding_view_size));
}

// TODO: add support for dr::mhp::halo(dr::mhp::views::sliding(r)).exchange()

} // namespace DR_RANGES_NAMESPACE
