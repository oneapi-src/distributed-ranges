// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/shp/algorithms/for_each.hpp>

namespace shp {

template <lib::distributed_range R, std::integral T> void iota(R &&r, T value) {
  auto iota_view = rng::views::iota(T(value), T(value + rng::distance(r)));

  shp::for_each(shp::par_unseq, shp::views::zip(iota_view, r), [](auto &&elem) {
    auto &&[idx, v] = elem;
    v = idx;
  });
}

template <lib::distributed_iterator Iter, std::integral T>
void iota(Iter begin, Iter end, T value) {
  auto r = rng::subrange(begin, end);
  iota(r, value);
}

} // namespace shp
