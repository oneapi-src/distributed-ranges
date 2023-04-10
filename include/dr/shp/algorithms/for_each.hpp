// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/util.hpp>
#include <dr/shp/zip_view.hpp>
#include <sycl/sycl.hpp>

namespace shp {

template <typename ExecutionPolicy, lib::distributed_range R, typename Fn>
void for_each(ExecutionPolicy &&policy, R &&r, Fn &&fn) {
  static_assert( // currently only one policy supported
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  std::vector<sycl::event> events;

  for (auto &&segment : lib::ranges::segments(r)) {
    auto &&q = __detail::queue(lib::ranges::rank(segment));

    assert(rng::distance(segment) > 0);

    auto local_segment = __detail::get_local_segment(segment);

    auto first = rng::begin(local_segment);

    auto event = q.parallel_for(rng::distance(local_segment),
                                [=](auto idx) { fn(*(first + idx)); });
    events.emplace_back(event);
  }
  __detail::wait(events);
}

template <typename ExecutionPolicy, lib::distributed_iterator Iter, typename Fn>
void for_each(ExecutionPolicy &&policy, Iter begin, Iter end, Fn &&fn) {
  for_each(std::forward<ExecutionPolicy>(policy), rng::subrange(begin, end),
           std::forward<Fn>(fn));
}

} // namespace shp
