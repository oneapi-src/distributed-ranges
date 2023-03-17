// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/distributed_span.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/util.hpp>
#include <dr/shp/zip_view.hpp>
#include <sycl/sycl.hpp>

namespace shp {
namespace __detail {

sycl::event for_each_async(const device_policy &policy,
                           lib::remote_range auto &&r, auto &&fn) {

  const std::size_t range_size = rng::distance(r);
  assert(range_size > 0);

  auto local_segment = __detail::get_local_segment(r);
  auto first = rng::begin(local_segment);

  auto device_of_rank = policy.get_devices()[lib::ranges::rank(r)];
  return sycl::queue(shp::context(), device_of_rank)
      .parallel_for(range_size, [=](auto idx) { fn(*(first + idx)); });
}
} // namespace __detail

template <typename ExecutionPolicy, lib::distributed_range R, typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {
  static_assert( // currently only one policy supported
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  std::vector<sycl::event> events;

  for (auto &&segment : lib::ranges::segments(r))
    events.emplace_back(__detail::for_each_async(policy, segment, fn));

  __detail::wait(events);
}

template <typename ExecutionPolicy, lib::distributed_iterator Iter, typename Fn>
void for_each(ExecutionPolicy &&policy, Iter begin, Iter end, Fn &&fn) {
  for_each(std::forward<ExecutionPolicy>(policy), rng::subrange(begin, end),
           std::forward<Fn>(fn));
}

} // namespace shp
