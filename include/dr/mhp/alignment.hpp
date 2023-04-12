// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/detail/ranges.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace mhp {

auto aligned(dr::distributed_range auto &&r) {
  return !dr::ranges::segments(r).empty();
}

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool aligned(dr::distributed_range auto &&r1, dr::distributed_range auto &&r2,
             dr::distributed_range auto &&...rest) {
  for (auto seg :
       rng::views::zip(dr::ranges::segments(r1), dr::ranges::segments(r2))) {
    if (dr::ranges::rank(seg.first) != dr::ranges::rank(seg.second)) {
      return false;
    }
    if (rng::distance(seg.first) != rng::distance(seg.second)) {
      return false;
    }
  }

  return aligned(r2, rest...);
}

template <typename T>
concept local_range = rng::forward_range<T> && !dr::distributed_range<T>;

// Skip local iterators
bool aligned(local_range auto &&r1, dr::distributed_range auto &&r2,
             dr::distributed_range auto... rest) {
  return aligned(r2, rest...);
}

bool aligned(dr::distributed_range auto &&r1, local_range auto &&r2,
             dr::distributed_range auto &&...rest) {
  return aligned(r1, rest...);
}

} // namespace mhp
