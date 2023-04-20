// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <limits>

#include <dr/concepts/concepts.hpp>
#include <dr/shp/algorithms/for_each.hpp>

namespace dr::shp {

namespace views {

struct iota_fn_ {
  template <std::integral W> auto operator()(W value) const {
    return rng::views::iota(value, std::numeric_limits<W>::max());
  }

  template <std::integral W, std::integral Bound>
  auto operator()(W value, Bound bound) {
    return rng::views::iota(value, W(bound));
  }
};

inline constexpr auto iota = iota_fn_{};

} // namespace views

template <dr::distributed_range R, std::integral T> void iota(R &&r, T value) {
  auto iota_view = rng::views::iota(value, T(value + rng::distance(r)));

  for_each(par_unseq, views::zip(iota_view, r), [](auto &&elem) {
    auto &&[idx, v] = elem;
    v = idx;
  });
}

template <dr::distributed_iterator Iter, std::integral T>
void iota(Iter begin, Iter end, T value) {
  auto r = rng::subrange(begin, end);
  iota(r, value);
}

} // namespace dr::shp
