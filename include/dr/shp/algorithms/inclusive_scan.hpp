// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>

#include <oneapi/dpl/execution>

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/vector.hpp>
#include <oneapi/dpl/async>
#include <oneapi/dpl/numeric>

#include <dr/concepts/concepts.hpp>

namespace {

// Precondition: std::distance(begin, end) >= 2
// Postcondition: inclusive scan performed on [begin, end), return future
template <typename ExecutionPolicy, std::forward_iterator Iter>
auto inclusive_scan_no_init_async(ExecutionPolicy &&policy, Iter begin,
                                  Iter end, Iter d_begin) {
  return oneapi::dpl::experimental::inclusive_scan_async(
      std::forward<ExecutionPolicy>(policy), begin, end, d_begin);
}

} // namespace

namespace shp {

template <typename ExecutionPolicy, lib::distributed_range R>
void inclusive_scan(ExecutionPolicy &&policy, R &&r) {
  namespace sycl = cl::sycl;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    /*
    using future_t = decltype(oneapi::dpl::experimental::reduce_async(
      oneapi::dpl::execution::device_policy(policy.get_devices()[0]),
      lib::ranges::segments(r)[0].begin(), lib::ranges::segments(r)[0].end(),
      lib::ranges::segments(r)[0].begin()));
      */

    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    // std::vector<future_t> futures;

    for (auto &&segment : lib::ranges::segments(r)) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(segment), rng::end(segment));

      if (dist >= 2) {
        auto future = inclusive_scan_no_init_async(
            local_policy, rng::begin(segment), rng::end(segment),
            rng::begin(segment));

        future.get();
        // futures.push_back(std::move(future));
      }
    }

    auto root = devices[0];
    auto &&segments = lib::ranges::segments(r);

    using VT = rng::range_value_t<R>;

    shp::device_allocator<VT> allocator(shp::context(), root);

    shp::vector<VT, shp::device_allocator<VT>> partial_sums(
        std::size_t(segments.size()), allocator);

    /*
        for (auto&& future : futures) {
          future.get();
        }
        */

    for (size_t i = 0; i < partial_sums.size(); i++) {
      auto &&segment = segments[i];

      auto iter = segment.end();
      --iter;

      VT v = *iter;

      partial_sums[i] = v;
    }

    for (size_t i = 1; i < partial_sums.size(); i++) {
      rng::range_value_t<R> v = partial_sums[i - 1];
      partial_sums[i] = v + partial_sums[i];
    }

    std::size_t idx = 0;
    for (auto &&segment : segments) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(segment), rng::end(segment));

      rng::range_value_t<R> sum = 0;

      if (idx > 0) {
        sum = partial_sums[idx - 1];
      }

      oneapi::dpl::experimental::for_each_async(
          local_policy, rng::begin(segment), rng::end(segment),
          [=](auto &&x) { x += sum; })
          .get();
      idx++;
    }

  } else {
    assert(false);
  }
}

} // namespace shp
