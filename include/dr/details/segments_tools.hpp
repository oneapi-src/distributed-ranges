// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace lib {

namespace internal {

// return number of full segments and remainder to cover n elements
template <typename R>
void n_segs_remainder(R &&segments, std::size_t n, auto &n_segs,
                      auto &remainder) {
  n_segs = 0;
  remainder = n;

  for (auto &&seg : segments) {
    if (seg.size() > remainder) {
      break;
    }
    remainder -= seg.size();
    n_segs++;
  }
}

template <typename R> auto enumerate(R &&segments) {
  return rng::views::zip(rng::views::iota(0u), segments);
}

// Take the first n elements
template <typename R> auto take_segments(R &&segments, std::size_t n) {
  std::size_t n_segs, remainder;
  n_segs_remainder(segments, n, n_segs, remainder);

  auto take_partial = [=](auto &&v) {
    auto &&[i, segment] = v;
    return rng::views::take(segment, i == n_segs ? remainder : segment.size());
  };

  return enumerate(segments) | rng::views::take(n_segs + 1) |
         rng::views::transform(take_partial);
}

// Drop the first n elements
template <typename R> auto drop_segments(R &&segments, std::size_t n) {
  std::size_t n_segs, remainder;
  n_segs_remainder(segments, n, n_segs, remainder);

  auto drop_partial = [=](auto &&v) {
    auto &&[i, segment] = v;
    return rng::views::drop(segment, i == n_segs ? remainder : 0);
  };

  return enumerate(segments) | rng::views::drop(n_segs) |
         rng::views::transform(drop_partial);
}

//
// Zip the segments for 1 or more distributed ranges. e.g.:
//
//   segments(dv1): [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
//   segments(dv2): [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
//
//   drop the first 4 elements and zip the segments for the rest
//
//    zip segments: [[(14, 24)], [(15, 25), (16, 26), (17, 27), (18, 28), (19,
//    29)]]
//
template <typename... Ss> auto zip_segments(Ss &&...iters) {
  auto zip_segment = [](auto &&v) {
    auto zip = [](auto &&...refs) { return rng::views::zip(refs...); };
    return std::apply(zip, v);
  };

  return rng::views::zip(lib::ranges::segments(iters)...) |
         rng::views::transform(zip_segment);
}

//
// Given an iter for a zip, return the segmentation
//
auto zip_iter_segments(auto zip_iter) {
  // Dereferencing a zip iterator returns a tuple of references, we
  // take the address of the references to iterators, and then get the
  // segments from the iterators.

  // Given the list of refs as arguments, convert to list of iters
  auto zip = [](auto &&...refs) { return zip_segments(&refs...); };

  // Convert the zip iterator to a tuple of references, and pass the
  // references as a list of arguments
  return std::apply(zip, *zip_iter);
}

} // namespace internal

} // namespace lib

namespace ranges {

#if 1
// A standard library range adaptor does not change the rank of a
// remote range, so we can simply return the rank of the base view.
template <rng::range V>
  requires(lib::remote_range<decltype(std::declval<V>().base())>)
auto rank_(V &&v) {
  return lib::ranges::rank(std::forward<V>(v).base());
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
  return take_segments(lib::ranges::segments(v.base()), v.size());
}

template <rng::range V>
  requires(lib::is_drop_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_range<decltype(std::declval<V>().base())>)
auto segments_(V &&v) {
  return drop_segments(lib::ranges::segments(v.base()),
                       v.base().size() - v.size());
}

template <rng::range V>
  requires(lib::is_subrange_view_v<std::remove_cvref_t<V>> &&
           lib::distributed_iterator<decltype(std::declval<V>().begin())>)
auto segments_(V &&v) {
  return take_segments(lib::ranges::segments(v.begin()), v.end() - v.begin());
}
#endif

template <rng::range... Views>
  requires(
      lib::is_zip_view_v<std::remove_cvref_t<Views>...> &&
      (lib::distributed_iterator<decltype(std::declval<Views>().begin())> &&
       ...))
auto segments_(rng::zip_view<Views...> &&zip) {
  //  return zip_iter_segments(zip.begin());
}

} // namespace ranges
