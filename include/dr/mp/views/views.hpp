// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/views/iota.hpp>
#include <dr/views/transform.hpp>

namespace dr::mp {

// Select segments local to this rank and convert the iterators in the
// segment to local
template <typename R> auto local_segments(R &&dr) {
  auto is_local = [](const auto &segment) {
    return dr::ranges::rank(segment) == default_comm().rank();
  };
  // Convert from remote iter to local iter
  auto local_iter = [](const auto &segment) {
    auto b = dr::ranges::local(rng::begin(segment));
    return rng::subrange(b, b + rng::distance(segment));
  };
  return dr::ranges::segments(std::forward<R>(dr)) |
         rng::views::filter(is_local) | rng::views::transform(local_iter);
}

template <typename R> auto local_segments_with_idx(R &&dr) {
  auto is_local = [](const auto &segment_with_idx) {
    return dr::ranges::rank(std::get<1>(segment_with_idx)) ==
           default_comm().rank();
  };
  // Convert from remote iter to local iter
  auto local_iter = [](const auto &segment_with_idx) {
    auto &&[idx, segment] = segment_with_idx;
    auto b = dr::ranges::local(rng::begin(segment));
    return std::tuple(idx, rng::subrange(b, b + rng::distance(segment)));
  };
  return dr::ranges::segments(std::forward<R>(dr)) | rng::views::enumerate |
         rng::views::filter(is_local) | rng::views::transform(local_iter);
}

template <dr::distributed_contiguous_range R> auto local_segment(R &&r) {
  auto segments = dr::mp::local_segments(std::forward<R>(r));

  if (rng::empty(segments)) {
    return rng::range_value_t<decltype(segments)>{};
  }

  // Should be error, not assert. Or we could join all the segments
  assert(rng::distance(segments) == 1);
  return *rng::begin(segments);
}

template <typename R> auto local_mdspans(R &&dr) {
  return dr::ranges::segments(std::forward<R>(dr))
         // Select the local segments
         | rng::views::filter([](auto s) {
             return dr::ranges::rank(s) == default_comm().rank();
           })
         // Extract the mdspan
         | rng::views::transform([](auto s) { return s.mdspan(); });
}

template <dr::distributed_contiguous_range R> auto local_mdspan(R &&r) {
  auto mdspans = dr::mp::local_mdspans(std::forward<R>(r));

  if (rng::empty(mdspans)) {
    return rng::range_value_t<decltype(mdspans)>{};
  }

  // Should be error, not assert. Or we could join all the segments
  assert(rng::distance(mdspans) == 1);
  return *rng::begin(mdspans);
}

} // namespace dr::mp

namespace dr::mp::views {

inline constexpr auto all = rng::views::all;
inline constexpr auto counted = rng::views::counted;
inline constexpr auto drop = rng::views::drop;
inline constexpr auto iota = dr::views::iota;
inline constexpr auto take = rng::views::take;
inline constexpr auto transform = dr::views::transform;

} // namespace dr::mp::views
