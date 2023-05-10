// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <limits>
#include <sycl/sycl.hpp>

namespace dr::shp {

namespace __detail {

template <typename Fn>
sycl::event parallel_for_64bit(sycl::queue &q, sycl::range<1> numWorkItems,
                               Fn &&fn) {
  std::size_t max_kernel_size = std::numeric_limits<std::int32_t>::max();

  std::vector<sycl::event> events;
  for (std::size_t base_idx = 0; base_idx < numWorkItems.size();
       base_idx += max_kernel_size) {
    std::size_t launch_size =
        std::min(numWorkItems.size() - base_idx, max_kernel_size);

    auto e = q.parallel_for(launch_size, [=](sycl::id<1> idx_) {
      sycl::id<1> idx(base_idx + idx_);
      fn(idx);
    });

    events.push_back(e);
  }

  auto e = q.submit([&](auto &&h) {
    h.depends_on(events);
    // Empty host task necessary due to [CMPLRLLVM-46542]
    h.host_task([] {});
  });

  return e;
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<1> numWorkItems, Fn &&fn) {
  std::size_t max_kernel_size = std::numeric_limits<std::int32_t>::max();

  if (numWorkItems.size() < max_kernel_size) {
    return q.parallel_for(numWorkItems, std::forward<Fn>(fn));
  } else {
    return parallel_for_64bit(q, numWorkItems, std::forward<Fn>(fn));
  }
}

} // namespace __detail

} // namespace dr::shp
