// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/distributed_span.hpp>
#include <dr/shp/zip_view.hpp>
#include <sycl/sycl.hpp>

namespace shp {

template <typename ExecutionPolicy, lib::distributed_contiguous_range R,
          typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {
  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    std::vector<sycl::queue> queues;
    std::vector<sycl::event> events;

    for (auto &&segment : lib::ranges::segments(r)) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(device);

      auto begin = lib::ranges::local(rng::begin(segment));

      assert(rng::size(segment) > 0);
      auto event = q.parallel_for(rng::size(segment),
                                  [=](auto idx) { fn(*(begin + idx)); });
      events.emplace_back(event);
      queues.emplace_back(q);
    }

    for (auto &&event : events) {
      event.wait();
    }
  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, lib::distributed_range R, typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    std::vector<sycl::queue> queues;
    std::vector<sycl::event> events;

    for (auto &&segment : lib::ranges::segments(r)) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(device);

      auto begin = rng::begin(segment);

      auto event = q.parallel_for(rng::size(segment),
                                  [=](auto idx) { fn(*(begin + idx)); });
      events.emplace_back(event);
      queues.emplace_back(q);
    }

    for (auto &&event : events) {
      event.wait();
    }
  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, lib::distributed_iterator Iter, typename Fn>
void for_each(ExecutionPolicy &&policy, Iter begin, Iter end, Fn &&fn) {
  for_each(std::forward<ExecutionPolicy>(policy), rng::subrange(begin, end),
           std::forward<Fn>(fn));
}

} // namespace shp
