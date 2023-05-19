// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <execution>
#include <type_traits>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

/// Collective iota on distributed range
template <dr::distributed_range R, std::integral T> void iota(R &&r, T value) {
  auto iota_view = rng::views::iota(value, T(value + rng::distance(r)));

  for_each(views::zip(iota_view, r), [](auto &&elem) {
    auto &&[idx, v] = elem;
    v = idx;
  });
}

/// Collective iota on iterator/sentinel for a distributed range
template <dr::distributed_iterator Iter, std::integral T>
void iota(Iter begin, Iter end, T value) {
  auto r = rng::subrange(begin, end);
  iota(r, value);
}

} // namespace dr::mhp
