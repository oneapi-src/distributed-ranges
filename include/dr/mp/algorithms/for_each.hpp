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
#include <dr/mp/global.hpp>

namespace dr::mp {

// the concept doesn't work yet... for some reason
template <typename R>
concept dual_vector_range =
    dr::distributed_range<R> && requires(R &r) { dr::ranges::segments(r)[0].is_compute(); };

void partial_for_each(dual_vector_range auto &&dr, auto op) {
  dr::drlog.debug(dr::logger::for_each, "partial_for_each: parallel execution\n");
  if (rng::empty(dr)) {
    return;
  }

  auto is_local = [](const auto &segment) {
    return dr::ranges::rank(segment) == default_comm().rank();
  };
  
  for (auto &seg : dr::ranges::segments(dr) | rng::views::filter(is_local)) {
    if (!seg.is_compute()) {
      seg.swap_state();
      continue;
    }

    auto b = dr::ranges::local(rng::begin(seg));
    auto s = rng::subrange(b, b + rng::distance(seg));

    if (mp::use_sycl()) {
      dr::drlog.debug("  using sycl\n");

      assert(rng::distance(s) > 0);
#ifdef SYCL_LANGUAGE_VERSION
      dr::__detail::parallel_for(
          dr::mp::sycl_queue(), sycl::range<1>(rng::distance(s)),
          [first = rng::begin(s), op](auto idx) { op(first[idx]); })
          .wait();
#else
      assert(false);
#endif
    } else {
      dr::drlog.debug("  using cpu\n");
      auto start = std::chrono::high_resolution_clock::now();
      rng::for_each(s, op);
      auto end = std::chrono::high_resolution_clock::now();
      auto length = duration_cast<std::chrono::nanoseconds>(end - start);
      dr::drlog.debug("  time spent: {}\n", length.count());
    }

    seg.swap_state();
  }
  barrier();
}

// Collective for_each on distributed range
void for_each(dr::distributed_range auto &&dr, auto op) {
  dr::drlog.debug(dr::logger::for_each, "for_each: parallel execution\n");
  if (rng::empty(dr)) {
    return;
  }
  assert(aligned(dr));

  for (const auto &s : local_segments(dr)) {
    if (mp::use_sycl()) {
      dr::drlog.debug("  using sycl\n");

      assert(rng::distance(s) > 0);
#ifdef SYCL_LANGUAGE_VERSION
      dr::__detail::parallel_for(
          dr::mp::sycl_queue(), sycl::range<1>(rng::distance(s)),
          [first = rng::begin(s), op](auto idx) { op(first[idx]); })
          .wait();
#else
      assert(false);
#endif
    } else {
      dr::drlog.debug("  using cpu\n");
      auto start = std::chrono::high_resolution_clock::now();
      rng::for_each(s, op);
      auto end = std::chrono::high_resolution_clock::now();
      auto length = duration_cast<std::chrono::nanoseconds>(end - start);
      dr::drlog.debug("  time spent: {}\n", length.count());
    }
  }
  barrier();
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mp::for_each(rng::subrange(first, last), op);
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI, std::integral I>
DI for_each_n(DI first, I n, auto op) {
  auto last = first;
  rng::advance(last, n);
  mp::for_each(first, last, op);
  return last;
}

} // namespace dr::mp
