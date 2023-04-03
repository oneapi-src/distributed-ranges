// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/details/ranges.hpp>
#include <dr/details/ranges_shim.hpp>

namespace mhp {

auto aligned(lib::distributed_range auto &&r) {
  return !lib::ranges::segments(r).empty();
}

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool aligned(lib::distributed_range auto &&r1, lib::distributed_range auto &&r2,
             lib::distributed_range auto &&...rest) {
  for (auto seg :
       rng::views::zip(lib::ranges::segments(r1), lib::ranges::segments(r2))) {
    if (lib::ranges::rank(seg.first) != lib::ranges::rank(seg.second)) {
      return false;
    }
    if (rng::distance(seg.first) != rng::distance(seg.second)) {
      return false;
    }
  }

  return aligned(r2, rest...);
}

template <typename T>
concept local_range = rng::forward_range<T> && !lib::distributed_range<T>;

// Skip local iterators
bool aligned(local_range auto &&r1, lib::distributed_range auto &&r2,
             lib::distributed_range auto... rest) {
  return aligned(r2, rest...);
}

bool aligned(lib::distributed_range auto &&r1, local_range auto &&r2,
             lib::distributed_range auto &&...rest) {
  return aligned(r1, rest...);
}

} // namespace mhp
