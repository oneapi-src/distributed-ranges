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

//
// Zip the segments for 1 or more distributed ranges. e.g.:
//
//   segments(dv1): [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
//   segments(dv2): [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
//
//   Assume we have dropped the first 4 elements of dv1 & dv2. Then we
//   zip them together and ask for the segments.
//
//    zip segments: [[(14, 24)], [(15, 25), (16, 26), (17, 27), (18, 28), (19,
//    29)]]
//
template <typename... Ss> auto zip_segments(Ss &&...iters) {
  auto zip_segment = [](auto &&v) {
    auto zip = [](auto &&...refs) { return rng::views::zip(refs...); };
    return std::apply(zip, v);
  };

  auto zipped = rng::views::zip(lib::ranges::segments(iters)...) |
                rng::views::transform(zip_segment);

  if (aligned(iters...)) {
    return zipped;
  } else {
    return decltype(zipped)();
  }
}

template <typename I>
concept is_zip_iterator =
    std::forward_iterator<I> && requires(I &iter) { std::get<0>(*iter); };

auto zip_iter_segments(is_zip_iterator auto zip_iter) {
  // Dereferencing a zip iterator returns a tuple of references, we
  // take the address of the references to iterators, and then get the
  // segments from the iterators.

  // Given the list of refs as arguments, convert to list of iters
  auto zip = [](auto &&...refs) { return zip_segments(&refs...); };

  // Convert the zip iterator to a tuple of references, and pass the
  // references as a list of arguments
  return std::apply(zip, *zip_iter);
}

auto zip_iter_rank(is_zip_iterator auto zip_iter) {
  return lib::ranges::rank(std::get<0>(*zip_iter));
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
    requires(R &segment) { lib::ranges::rank(&(std::get<0>(segment[0]))); };

template <typename R>
concept distributed_zip_reference = requires(R &&reference) {
                                      {
                                        &std::get<0>(reference)
                                        } -> lib::distributed_iterator;
                                    };

template <zip_segment Segment> auto rank_(Segment &&segment) {
  return lib::ranges::rank(&(std::get<0>(segment[0])));
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

template <rng::range V>
  requires(lib::is_zip_view_v<std::remove_cvref_t<V>> &&
           distributed_zip_reference<rng::range_reference_t<V>>)
auto segments_(V &&zip) {
  return take_segments(lib::internal::zip_iter_segments(zip.begin()),
                       rng::distance(rng::begin(zip), rng::end(zip)));
}

template <lib::internal::is_zip_iterator ZI> auto segments_(ZI zi) {
  return lib::internal::zip_iter_segments(zi);
}

template <lib::internal::is_zip_iterator ZI> auto local_(ZI zi) {
  auto refs_to_local_zip_iterator = [](auto &&...refs) {
    // Convert the first segment of each component to local and then
    // zip them together, returning the begin() of the zip view
    return rng::zip_view(
               (lib::ranges::local(lib::ranges::segments(&refs)[0]))...)
        .begin();
  };
  return std::apply(refs_to_local_zip_iterator, *zi);
}

} // namespace DR_RANGES_NAMESPACE
