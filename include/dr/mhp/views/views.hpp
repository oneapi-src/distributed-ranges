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
  return dr::ranges::segments(std::forward<R>(dr)) |
         rng::views::filter(is_local) |
         rng::views::transform(
             [](const auto &segment) {
               int x [[maybe_unused]] = 1;
               return dr::ranges::local(segment);
             });
}

template <dr::distributed_contiguous_range R> auto local_segment(R &&r) {
  auto segments = dr::mhp::local_segments(std::forward<R>(r));

  // Should be error, not assert
  assert(rng::distance(segments) == 1);
  return *rng::begin(segments);
}

} // namespace dr::mhp

namespace dr::mhp::views {

inline constexpr auto all = rng::views::all;
inline constexpr auto counted = rng::views::counted;
inline constexpr auto drop = rng::views::drop;
inline constexpr auto iota = dr::views::iota;
inline constexpr auto take = rng::views::take;
inline constexpr auto transform = dr::views::transform;

} // namespace dr::mhp::views
