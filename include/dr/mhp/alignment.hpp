// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

namespace mhp {

// It won't have segments if it is a zip of non-aligned data
bool aligned(lib::distributed_iterator auto iter) {
  return !lib::ranges::segments(iter).empty();
}

// iter1 is aligned with iter2, and iter2 is aligned with the rest
bool aligned(lib::distributed_iterator auto iter1,
             lib::distributed_iterator auto iter2,
             lib::distributed_iterator auto... iters) {
  auto combined = rng::views::zip(lib::ranges::segments(iter1),
                                  lib::ranges::segments(iter2));
  if (combined.empty())
    return false;
  for (auto seg : combined) {
    if (lib::ranges::rank(seg.first) != lib::ranges::rank(seg.second) ||
        rng::distance(seg.first) != rng::distance(seg.second)) {
      return false;
    }
  }

  return aligned(iter2, iters...);
}

} // namespace mhp
