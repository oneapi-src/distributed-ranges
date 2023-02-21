// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>
#include <dr/concepts/concepts.hpp>
#include <dr/details/ranges_shim.hpp>
#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/distributed_span.hpp>
#include <dr/shp/zip_view.hpp>

namespace shp {

template <lib::distributed_contiguous_range R, typename Fn>
void for_each(const device_policy &policy, R &&r, Fn &&fn) {
  std::span<const cl::sycl::device> devices = policy.get_devices();

  std::vector<cl::sycl::queue> queues;
  std::vector<cl::sycl::event> events;

  for (auto &&segment : lib::ranges::segments(r)) {

    sycl::queue q(devices[lib::ranges::rank(segment)]);

    auto begin = rng::begin(segment);

    assert(rng::size(segment) > 0);
    auto event = q.parallel_for(
        rng::size(segment), [=](cl::sycl::id<1> idx) { fn(*(begin + idx)); });
    events.emplace_back(event);
    queues.emplace_back(q);
  }

  for (auto &&event : events) {
    event.wait();
  }
}

template <lib::distributed_iterator Iter, typename Fn>
void for_each(const device_policy &policy, Iter begin, Iter end, Fn &&fn) {
  for_each(policy, rng::subrange(begin, end), std::forward<Fn>(fn));
}

} // namespace shp
