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

class enumerate_adapter_closure {
public:
  enumerate_adapter_closure() {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    return enumerate(std::forward<R>(r));
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const enumerate_adapter_closure &closure) {
    return enumerate(std::forward<R>(r));
  }
};

template <rng::viewable_range R> auto enumerate(R &&r) {
  using W = rng::range_size_t<R>;
  return rng::views::zip(rng::views::iota(W{0}), std::forward<R>(r));
}

inline auto enumerate() { return enumerate_adapter_closure(); }

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

} // namespace internal

} // namespace lib

namespace ranges {

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
  auto begin = rng::begin(v);
  auto end = rng::begin(v);

  auto seg_begin = lib::ranges::segment_index(begin);
  auto local_begin = lib::ranges::local_index(begin);

  auto seg_end = lib::ranges::segment_index(end);
  auto local_end = lib::ranges::local_index(end);

  auto n_segs = seg_end - seg_begin;

  return lib::ranges::segments(begin) | lib::internal::enumerate() |
         rng::views::drop(seg_begin) | rng::views::transform([=](auto &&e) {
           auto &&[i, seg] = e;
           if (i == seg_begin) {
             return seg | rng::views::drop(local_begin);
           } else {
             return seg | rng::views::drop(0);
           }
         }) |
         rng::views::take(n_segs) | lib::internal::enumerate() |
         rng::views::transform([=](auto &&e) {
           auto &&[i, seg] = e;
           if (i == n_segs - 1) {
             return seg | rng::views::take(local_end);
           } else {
             return seg | rng::views::take(0);
           }
         });

  /*
    return lib::internal::take_segments(lib::ranges::segments(v.begin()),
                                        v.end() - v.begin());
                                        */
}

} // namespace ranges
