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
#include <dr/detail/sycl_utils.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

/// Collective for_each on distributed range
void for_each(dr::distributed_range auto &&dr, auto op) {
  dr::drlog.debug(dr::logger::for_each, "for_each: parallel execution\n");
  if (rng::empty(dr)) {
    return;
  }
  assert(aligned(dr));

  for (const auto &s : local_segments(dr)) {
    if (mhp::use_sycl()) {
      dr::drlog.debug("  using sycl\n");

      assert(rng::distance(s) > 0);
#ifdef SYCL_LANGUAGE_VERSION
      dr::__detail::parallel_for(
          dr::mhp::sycl_queue(), sycl::range<1>(rng::distance(s)),
          [first = rng::begin(s), op](auto idx) { op(first[idx]); })
          .wait();
#else
      assert(false);
#endif
    } else {
      dr::drlog.debug("  using cpu\n");
      rng::for_each(s, op);
    }
  }
  barrier();
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mhp::for_each(rng::subrange(first, last), op);
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI, std::integral I>
DI for_each_n(DI first, I n, auto op) {
  auto last = first;
  rng::advance(last, n);
  mhp::for_each(first, last, op);
  return last;
}

} // namespace dr::mhp
