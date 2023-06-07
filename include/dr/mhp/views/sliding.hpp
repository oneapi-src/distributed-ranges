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

  auto base_segments = dr::ranges::segments(v.base());
  auto elements_to_skip_in_base = rng::size(v.base());
  auto elements_to_take = 0;
  if (!rng::empty(v)) {
    // need to reverse engineer `n` which was passed to sliding_view
    elements_to_take = rng::size(v);
    const auto slide_size = elements_to_skip_in_base - elements_to_take + 1;
    // TODO: this code assumes that halo is symmetric, thus odd (center + 2n)
    // note, it is not an assertion preventing all wrong use cases
    // other ones are caught by assert during attempt to read outside halo
    assert(slide_size % 2 == 1);
    elements_to_skip_in_base = slide_size / 2;
  }

  return dr::mhp::views::segmented(
      v, dr::__detail::take_segments(
          dr::__detail::drop_segments(base_segments, elements_to_skip_in_base), elements_to_take));
}

// TODO: add support for dr::mhp::halo(dr::mhp::views::sliding(r)).exchange()
} // namespace DR_RANGES_NAMESPACE
