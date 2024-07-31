// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/algorithms/inclusive_exclusive_scan_impl.hpp>

namespace dr::mp {

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T, typename BinaryOp>
auto exclusive_scan(R &&r, O &&o, T init, BinaryOp &&binary_op) {
  return __detail::inclusive_exclusive_scan_impl_<true>(
      std::forward<R>(r), rng::begin(std::forward<O>(o)),
      std::forward<BinaryOp>(binary_op),
      std::optional<rng::range_value_t<R>>(init));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T>
auto exclusive_scan(R &&r, O &&o, T init) {
  return dr::mp::exclusive_scan(std::forward<R>(r), std::forward<O>(o),
                                static_cast<rng::range_value_t<R>>(init),
                                std::plus<rng::range_value_t<R>>());
}

// Distributed iterator versions

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename T, typename BinaryOp>
OutputIter exclusive_scan(Iter first, Iter last, OutputIter d_first, T init,
                          BinaryOp &&binary_op) {

  return dr::mp::exclusive_scan(rng::subrange(first, last), d_first,
                                std::forward<BinaryOp>(binary_op), init);
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename T>
OutputIter exclusive_scan(Iter first, Iter last, OutputIter d_first, T init) {
  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  dr::mp::exclusive_scan(rng::subrange(first, last),
                         rng::subrange(d_first, d_last), init);

  return d_last;
}

} // namespace dr::mp
