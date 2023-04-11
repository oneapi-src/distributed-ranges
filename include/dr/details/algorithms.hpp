// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// ALGORITHMS CODE SHARED BETWEEN shp AND mhp
// DO NOT INCLUDE THIS FILE DIRECTLY
// INCLUDED BY algorithms.hpp FILES IN mhp AND shp

template <lib::distributed_range R, std::integral T> void iota(R &&r, T value) {
  auto iota_view = rng::views::iota(value, T(value + rng::distance(r)));

  for_each(par_unseq, views::zip(iota_view, r), [](auto &&elem) {
    auto &&[idx, v] = elem;
    v = idx;
  });
}

template <lib::distributed_iterator Iter, std::integral T>
void iota(Iter begin, Iter end, T value) {
  auto r = rng::subrange(begin, end);
  iota(r, value);
}
