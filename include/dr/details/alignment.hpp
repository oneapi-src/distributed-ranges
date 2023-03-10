// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "dr/concepts/concepts.hpp"

namespace lib {
namespace __details {

// It won't have segments if it is a zip of non-aligned data
bool aligned(distributed_iterator auto iter) {
  return !ranges::segments(iter).empty();
}

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool aligned(distributed_iterator auto iter1, distributed_iterator auto iter2,
             distributed_iterator auto... iters) {
  auto combined =
      rng::views::zip(ranges::segments(iter1), ranges::segments(iter2));
  if (combined.empty())
    return false;
  for (auto seg : combined) {
    if (ranges::rank(seg.first) != ranges::rank(seg.second) ||
        seg.first.size() != seg.second.size()) {
      return false;
    }
  }

  return aligned(iter2, iters...);
}

} // namespace __details
} // namespace lib
