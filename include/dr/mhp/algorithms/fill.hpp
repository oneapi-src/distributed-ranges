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

/// Collective fill on distributed range
auto fill(dr::distributed_contiguous_range auto &&dr, auto value) {
  for_each(dr, [=](auto &v) { v = value; });
  return rng::end(dr);
}

/// Collective fill on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
auto fill(DI first, DI last, auto value) {
  mhp::fill(rng::subrange(first, last), value);
  return last;
}

} // namespace dr::mhp
