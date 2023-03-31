// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/details/ranges.hpp>
#include <dr/details/ranges_shim.hpp>

namespace mhp {

auto aligned(auto &&r) { return !lib::ranges::segments(r).empty(); }

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool aligned(lib::distributed_iterator auto iter1,
             lib::distributed_iterator auto iter2,
             lib::distributed_iterator auto... iters) {
  for (auto seg : rng::views::zip(lib::ranges::segments(iter1),
                                  lib::ranges::segments(iter2))) {
    if (lib::ranges::rank(seg.first) != lib::ranges::rank(seg.second)) {
      return false;
    }
    if (rng::distance(seg.first) != rng::distance(seg.second)) {
      return false;
    }
  }

  return aligned(iter2, iters...);
}

template <typename T>
concept local_iterator = !lib::distributed_iterator<T>;

// Skip local iterators
bool aligned(local_iterator auto iter1, lib::distributed_iterator auto iter2,
             lib::distributed_iterator auto... iters) {
  return aligned(iter2, iters...);
}

bool aligned(lib::distributed_iterator auto iter1, local_iterator auto iter2,
             lib::distributed_iterator auto... iters) {
  return aligned(iter1, iters...);
}

} // namespace mhp
