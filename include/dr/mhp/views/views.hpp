// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/views/iota.hpp>
#include <dr/views/transform.hpp>

namespace dr::mhp {

// Select segments local to this rank and convert the iterators in the
// segment to local
template <typename R> auto local_segments(R &&dr) {
  auto is_local = [](const auto &segment) {
    return dr::ranges::rank(segment) == default_comm().rank();
  };

  // Convert from remote iter to local iter
  auto local_iter = [](const auto &segment) {
    if constexpr (is_sliding_view_v<std::remove_cvref_t<R>>) {
      return segment | rng::views::transform(
                           [](const auto &&v) { return dr::ranges::local(v); });
    } else {
      auto b = dr::ranges::local(rng::begin(segment));
      return rng::subrange(b, b + rng::distance(segment));
    }
  };
  return dr::ranges::segments(std::forward<R>(dr)) |
         rng::views::filter(is_local) | rng::views::transform(local_iter);
}

} // namespace dr::mhp

namespace dr::mhp::views {

inline constexpr auto all = rng::views::all;
inline constexpr auto drop = rng::views::drop;
inline constexpr auto iota = dr::views::iota;
inline constexpr auto take = rng::views::take;
inline constexpr auto transform = dr::views::transform;

} // namespace dr::mhp::views
