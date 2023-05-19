// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/detail/ranges.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr::mhp {

template <typename T>
concept has_segments = requires(T &t) { dr::ranges::segments(t); };

template <typename T>
concept no_segments = !has_segments<T>;

auto sub_aligned(bool non_empty, has_segments auto &&r) {
  if (non_empty && dr::ranges::segments(r).empty()) {
    dr::drlog.debug("unaligned: empty segments\n");
    return false;
  } else {
    return true;
  }
}

auto sub_aligned(bool non_empty, auto &&r) { return true; }

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool sub_aligned(bool non_empty, has_segments auto &&r1, has_segments auto &&r2,
                 auto &&...rest) {
  auto z = rng::views::zip(dr::ranges::segments(r1), dr::ranges::segments(r2));
  auto i = rng::distance(z) - 1;
  for (auto seg : z) {
    if (dr::ranges::rank(seg.first) != dr::ranges::rank(seg.second)) {
      dr::drlog.debug("unaligned: ranks: {} {}\n", dr::ranges::rank(seg.first),
                      dr::ranges::rank(seg.second));
      return false;
    }
    // Size mismatch would misalign following segments. Skip test if this is the
    // last segment
    if (i > 0 && rng::distance(seg.first) != rng::distance(seg.second)) {
      dr::drlog.debug("unaligned: size: {} {}\n", rng::distance(seg.first),
                      rng::distance(seg.second));
      return false;
    }
    i--;
  }

  if (!rng::empty(dr::ranges::segments(r1))) {
    non_empty = true;
  }
  return sub_aligned(non_empty, r2, rest...);
}

// Skip local iterators
bool sub_aligned(bool non_empty, no_segments auto &&r1, has_segments auto &&r2,
                 auto... rest) {
  return sub_aligned(non_empty, r2, rest...);
}

bool sub_aligned(bool non_empty, has_segments auto &&r1, no_segments auto &&r2,
                 auto &&...rest) {
  return sub_aligned(non_empty, r1, rest...);
}

template <typename... Args> bool aligned(Args &&...args) {
  return sub_aligned(false, std::forward<Args>(args)...);
}

} // namespace dr::mhp
