// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <dr/shp/algorithms/inclusive_scan.hpp>

namespace dr::shp {

namespace __detail {

template <typename T> struct inverse;

template <typename T> struct inverse<std::plus<T>> {
  auto operator()() const { return std::minus<T>{}; }
};

template <typename T> struct inverse<std::multiplies<T>> {
  auto operator()() const { return std::divides<T>{}; }
};

} // namespace __detail

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename U, typename BinaryOp>
void exclusive_scan_impl_(ExecutionPolicy &&policy, R &&r, O &&o, U init,
                          BinaryOp &&binary_op) {
  inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o),
                       std::forward<BinaryOp>(binary_op), std::optional(init));

  auto elems = rng::subrange(rng::next(rng::begin(o)), rng::end(o));
  auto elems_sub = rng::subrange(rng::begin(o), rng::prev(rng::end(o)));

  auto inverse_op = __detail::inverse<std::remove_cvref_t<BinaryOp>>{}();

  auto z = views::zip(elems, elems_sub);

  for_each(shp::par_unseq, z, [=](auto &&e) {
    auto &&[a, b] = e;
    a = inverse_op(a, inverse_op(a, b));
  });

  *rng::begin(o) = init;
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T, typename BinaryOp>
void exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init,
                    BinaryOp &&binary_op) {
  exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o), init,
                       std::forward<BinaryOp>(binary_op));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T>
void exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init) {
  exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o), init,
                       std::plus<>{});
}

} // namespace dr::shp
