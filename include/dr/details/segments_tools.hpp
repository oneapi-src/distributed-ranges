// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/details/enumerate.hpp>
#include <dr/details/ranges_shim.hpp>
#include <dr/details/remote_subrange.hpp>
#include <dr/details/view_detectors.hpp>

namespace lib {

namespace internal {

// count the number of segments necessary to cover n elements,
// returning the index of the last segment and its remainder
template <typename R>
void n_segs_remainder(R &&segments, std::size_t n, auto &last_seg,
                      auto &remainder) {
  last_seg = 0;
  remainder = n;

  for (auto &&seg : segments) {
    if (seg.size() >= remainder) {
      break;
    }
    remainder -= seg.size();

    last_seg++;
  }
}

// Take all elements up to and including segment `segment_id` at index
// `local_id`
template <typename R>
auto take_segments(R &&segments, std::size_t segment_id, std::size_t local_id) {
  auto last_seg = segment_id;
  auto remainder = local_id;

  auto take_partial = [=](auto &&v) {
    auto &&[i, segment] = v;
    if (i == last_seg) {
      auto first = rng::begin(segment);
      auto last = rng::begin(segment);
      rng::advance(last, remainder);
      return lib::remote_subrange(first, last, lib::ranges::rank(segment));
    } else {
      return lib::remote_subrange(segment);
    }
  };

  return enumerate(segments) | rng::views::take(last_seg + 1) |
         rng::views::transform(std::move(take_partial));
}

// Take the first n elements
template <typename R> auto take_segments(R &&segments, std::size_t n) {
  std::size_t last_seg, remainder;
  n_segs_remainder(segments, n, last_seg, remainder);

  return take_segments(std::forward<R>(segments), last_seg, remainder);
}

// Drop all elements up to segment `segment_id` and index `local_id`
template <typename R>
auto drop_segments(R &&segments, std::size_t segment_id, std::size_t local_id) {
  auto last_seg = segment_id;
  auto remainder = local_id;

  auto drop_partial = [=](auto &&v) {
    auto &&[i, segment] = v;
    if (i == last_seg) {
      auto first = rng::begin(segment);
      rng::advance(first, remainder);
      auto last = rng::end(segment);
      return lib::remote_subrange(first, last, lib::ranges::rank(segment));
    } else {
      return lib::remote_subrange(segment);
    }
  };

  return enumerate(segments) | rng::views::drop(last_seg) |
         rng::views::transform(std::move(drop_partial));
}

// Drop the first n elements
template <typename R> auto drop_segments(R &&segments, std::size_t n) {
  std::size_t last_seg, remainder;
  n_segs_remainder(segments, n, last_seg, remainder);

  return drop_segments(std::forward<R>(segments), last_seg, remainder);
}

} // namespace internal

} // namespace lib

namespace DR_RANGES_NAMESPACE {

// A standard library range adaptor does not change the rank of a
// remote range, so we can simply return the rank of the base view.
template <rng::range V>
  requires(lib::remote_range<decltype(std::declval<V>().base())>)
auto rank_(V &&v) {
  return lib::ranges::rank(std::forward<V>(v).base());
}

template <typename R>
concept zip_segment =
    requires(R &segment) { lib::ranges::rank(&(std::get<1>(segment[0]))); };

template <zip_segment Segment> auto rank_(Segment &&segment) {
  return lib::ranges::rank(&(std::get<1>(segment[0])));
}

template <rng::range V>
  requires(lib::is_ref_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return lib::ranges::segments(v.base());
}

template <rng::range V>
  requires(lib::is_take_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return lib::internal::take_segments(lib::ranges::segments(v.base()),
                                      v.size());
}

template <rng::range V>
  requires(lib::is_drop_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return lib::internal::drop_segments(lib::ranges::segments(v.base()),
                                      v.base().size() - v.size());
}

template <rng::range V>
  requires(lib::is_subrange_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_iterator<decltype(std::declval<V>().begin())>)
auto segments_(V &&v) {
  auto first = rng::begin(v);
  auto last = rng::end(v);

  auto size = rng::distance(first, last);

  return lib::internal::take_segments(lib::ranges::segments(first), size);
}

} // namespace DR_RANGES_NAMESPACE
