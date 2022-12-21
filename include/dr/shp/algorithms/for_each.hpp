// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>
#include <concepts/concepts.hpp>
#include <shp/algorithms/execution_policy.hpp>
#include <shp/distributed_span.hpp>
#include <shp/zip_view.hpp>

namespace shp {

template <typename ExecutionPolicy, lib::distributed_contiguous_range R,
          typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {

  namespace sycl = cl::sycl;

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

      auto begin = lib::ranges::local(std::ranges::begin(segment));

      auto event = q.parallel_for(sycl::range<1>(std::ranges::size(segment)),
                                  [=](sycl::id<1> idx) { fn(*(begin + idx)); });
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

  namespace sycl = cl::sycl;

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

      auto begin = std::ranges::begin(segment);

      auto event = q.parallel_for(sycl::range<1>(std::ranges::size(segment)),
                                  [=](sycl::id<1> idx) { fn(*(begin + idx)); });
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

} // namespace shp
