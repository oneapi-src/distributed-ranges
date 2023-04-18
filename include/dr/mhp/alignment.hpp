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

auto aligned(has_segments auto &&r) { return !dr::ranges::segments(r).empty(); }

auto aligned(auto &&r) { return true; }

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool aligned(has_segments auto &&r1, has_segments auto &&r2, auto &&...rest) {
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

  return aligned(r2, rest...);
}

// Skip local iterators
bool aligned(no_segments auto &&r1, has_segments auto &&r2, auto... rest) {
  return aligned(r2, rest...);
}

bool aligned(has_segments auto &&r1, no_segments auto &&r2, auto &&...rest) {
  return aligned(r1, rest...);
}

} // namespace dr::mhp
