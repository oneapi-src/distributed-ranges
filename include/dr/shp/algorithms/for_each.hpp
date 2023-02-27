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

template <typename ExecutionPolicy, lib::distributed_contiguous_range R,
          typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {
  static_assert( // currently only one policy supported
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  auto devices = policy.get_devices();
  std::vector<cl::sycl::event> events;

  for (auto &&segment : lib::ranges::segments(r)) {
    assert(rng::size(segment) > 0);
    events.emplace_back(
        sycl::queue(devices[lib::ranges::rank(segment)])
            .parallel_for(rng::size(segment), [=](cl::sycl::id<1> idx) {
              fn(*(rng::begin(segment) + idx));
            }));
  }

  for (auto &&event : events) {
    event.wait();
  }
}

template <typename ExecutionPolicy, lib::distributed_iterator Iter, typename Fn>
void for_each(ExecutionPolicy &&policy, Iter begin, Iter end, Fn &&fn) {
  for_each(std::forward<ExecutionPolicy>(policy), rng::subrange(begin, end),
           std::forward<Fn>(fn));
}

} // namespace shp
