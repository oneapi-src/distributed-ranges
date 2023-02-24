// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>

#include <oneapi/dpl/execution>

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/init.hpp>
#include <oneapi/dpl/async>
#include <oneapi/dpl/numeric>

#include <dr/concepts/concepts.hpp>

namespace {

// Precondition: std::distance(begin, end) >= 2
// Postcondition: return future to [begin, end) reduced with fn
template <typename T, typename ExecutionPolicy,
          std::bidirectional_iterator Iter, typename Fn>
auto reduce_no_init_async(ExecutionPolicy &&policy, Iter begin, Iter end,
                          Fn &&fn) {
  Iter new_end = end;
  --new_end;

  std::iter_value_t<Iter> init = *new_end;

  return oneapi::dpl::experimental::reduce_async(
      std::forward<ExecutionPolicy>(policy), begin, new_end,
      static_cast<T>(init), std::forward<Fn>(fn));
}

} // namespace

namespace shp {

template <typename ExecutionPolicy, lib::distributed_range R, typename T,
          typename BinaryOp>
T reduce(ExecutionPolicy &&policy, R &&r, T init, BinaryOp &&binary_op) {

  namespace sycl = cl::sycl;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    using future_t = decltype(oneapi::dpl::experimental::reduce_async(
        oneapi::dpl::execution::device_policy(policy.get_devices()[0]),
        lib::ranges::segments(r)[0].begin(), lib::ranges::segments(r)[0].end(),
        init, std::forward<BinaryOp>(binary_op)));

    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    std::vector<future_t> futures;

    for (auto &&segment : lib::ranges::segments(r)) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(segment), rng::end(segment));
      if (dist <= 0) {
        continue;
      } else if (dist == 1) {
        init = std::forward<BinaryOp>(binary_op)(init, *rng::begin(segment));
        continue;
      }

      auto future = reduce_no_init_async<T>(local_policy, rng::begin(segment),
                                            rng::end(segment),
                                            std::forward<BinaryOp>(binary_op));

      futures.push_back(std::move(future));
    }

    for (auto &&future : futures) {
      init = std::forward<BinaryOp>(binary_op)(init, future.get());
    }
    return init;
  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, lib::distributed_range R, typename T>
T reduce(ExecutionPolicy &&policy, R &&r, T init) {
  return reduce(std::forward<ExecutionPolicy>(policy), std::forward<R>(r), init,
                std::plus<>());
}

template <typename ExecutionPolicy, lib::distributed_range R>
rng::range_value_t<R> reduce(ExecutionPolicy &&policy, R &&r) {
  return reduce(std::forward<ExecutionPolicy>(policy), std::forward<R>(r),
                rng::range_value_t<R>{}, std::plus<>());
}

// Iterator versions

template <typename ExecutionPolicy, lib::distributed_iterator Iter>
std::iter_value_t<Iter> reduce(ExecutionPolicy &&policy, Iter begin, Iter end) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(begin, end), std::iter_value_t<Iter>{},
                std::plus<>());
}

template <typename ExecutionPolicy, lib::distributed_iterator Iter, typename T>
T reduce(ExecutionPolicy &&policy, Iter begin, Iter end, T init) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(begin, end), init, std::plus<>());
}

template <typename ExecutionPolicy, lib::distributed_iterator Iter, typename T,
          typename BinaryOp>
T reduce(ExecutionPolicy &&policy, Iter begin, Iter end, T init,
         BinaryOp &&binary_op) {
  return reduce(std::forward<ExecutionPolicy>(policy),
                rng::subrange(begin, end), init,
                std::forward<BinaryOp>(binary_op));
}

} // namespace shp
