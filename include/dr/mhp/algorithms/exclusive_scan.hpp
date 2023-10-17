// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/async>
#endif

#include <dr/detail/sycl_utils.hpp>
#include <dr/mhp/algorithms/inclusive_exclusive_scan_impl.hpp>

namespace dr::mhp {

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T, typename BinaryOp>
auto exclusive_scan(R &&r, O &&o, T init, BinaryOp &&binary_op) {
  return __detail::inclusive_exclusive_scan_impl_<true>(
      std::forward<R>(r), rng::begin(std::forward<O>(o)),
      std::forward<BinaryOp>(binary_op), std::optional(init));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T>
auto exclusive_scan(R &&r, O &&o, T init) {
  return dr::mhp::exclusive_scan(std::forward<R>(r), std::forward<O>(o), init,
                                 std::plus<rng::range_value_t<R>>());
}

// Distributed iterator versions

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename T, typename BinaryOp>
OutputIter exclusive_scan(Iter first, Iter last, OutputIter d_first, T init,
                          BinaryOp &&binary_op) {

  return dr::mhp::exclusive_scan(rng::subrange(first, last), d_first,
                                 std::forward<BinaryOp>(binary_op), init);
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename T>
OutputIter exclusive_scan(Iter first, Iter last, OutputIter d_first, T init) {
  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  dr::mhp::exclusive_scan(rng::subrange(first, last),
                          rng::subrange(d_first, d_last), init);

  return d_last;
}

} // namespace dr::mhp
