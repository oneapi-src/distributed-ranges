// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts/concepts.hpp>
#include <ranges>
#include <shp/zip_view.hpp>

namespace lib {

namespace internal {

// Trim the total number of elements included in all the
// ranges in `segments` to `n`.
template <std::ranges::random_access_range R>
auto trim_segments(R &&segments, std::size_t n) {
  std::size_t n_segs = 0;
  std::size_t count = 0;

  std::size_t last_segment_size;

  for (auto &&seg : segments) {
    n_segs++;
    count += seg.size();
    if (count >= n) {
      last_segment_size = seg.size() - (count - n);
      break;
    }
  }

  auto new_segments = shp::zip_view(
      std::ranges::views::iota(int(0), int(n_segs)),
      std::ranges::views::take(std::forward<R>(segments), n_segs));

  return std::ranges::views::transform(new_segments, [=](auto &&s) {
    auto &&[i, v] = s;
    if (i == n_segs - 1) {
      return std::ranges::views::take(v, last_segment_size);
    } else {
      return std::ranges::views::take(v, v.size());
    }
  });
}

struct drop {
  std::size_t remainder;
  template <typename R> auto operator()(R &&s) {
    auto v = std::ranges::views::drop(std::forward<R>(s), remainder);
    remainder = 0;
    return v;
  }
};

// Drop the first n elements
template <std::ranges::random_access_range R>
auto drop_segments(R &&segments, std::size_t n) {
  std::size_t n_segs_dropped = 0;
  std::size_t remainder = n;

  for (auto &&seg : segments) {
    if (seg.size() > remainder) {
      break;
    }
    remainder -= seg.size();
    n_segs_dropped++;
  }

  return std::forward<R>(segments) | std::ranges::views::drop(n_segs_dropped) |
         std::ranges::views::transform(drop{remainder});
}

} // namespace internal

} // namespace lib

namespace std {

namespace ranges {

// A standard library range adaptor does not change the rank of a
// remote range, so we can simply return the rank of the base view.
template <std::ranges::range V>
  requires(lib::remote_range<decltype(std::declval<V>().base())>)
auto rank_(V &&v) {
  return lib::ranges::rank(std::forward<V>(v).base());
}

template <std::ranges::range V>
  requires(lib::is_ref_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return lib::ranges::segments(v.base());
}

template <std::ranges::range V>
  requires(lib::is_take_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return lib::internal::trim_segments(lib::ranges::segments(v.base()),
                                      v.size());
}

template <std::ranges::range V>
  requires(lib::is_drop_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return lib::internal::drop_segments(lib::ranges::segments(v.base()),
                                      v.base().size() - v.size());
}

} // namespace ranges

} // namespace std
