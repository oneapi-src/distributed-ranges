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

/// Collective for_each on distributed range
void for_each(dr::distributed_range auto &&dr, auto op) {
  if (aligned(dr)) {
    dr::drlog.debug("for_each: parallel execution\n");
    for (const auto &s : local_segments(dr)) {
      if (mhp::use_sycl()) {
        std::for_each(dpl_policy(),
                      dr::__detail::direct_iterator(rng::begin(s)),
                      dr::__detail::direct_iterator(rng::end(s)), op);
      } else {
        rng::for_each(s, op);
      }
    }
  } else {
    rng::for_each(dr, op);
  }

  barrier();
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mhp::for_each(rng::subrange(first, last), op);
}

} // namespace dr::mhp
