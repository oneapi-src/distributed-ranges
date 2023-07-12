// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <limits>
#include <sycl/sycl.hpp>

namespace dr::shp {

namespace __detail {

// With the ND-range workaround, the maximum kernel size is
// `std::numeric_limits<std::int32_t>::max()` rounded down to
// the nearest multiple of the block size.
inline std::size_t max_kernel_size_(std::size_t block_size = 128) {
  std::size_t max_kernel_size = std::numeric_limits<std::int32_t>::max();
  return (max_kernel_size / block_size) * block_size;
}

// This is a workaround to avoid performance degradation
// in DPC++ for odd range sizes.
template <typename Fn>
sycl::event parallel_for_workaround(sycl::queue &q, sycl::range<1> numWorkItems,
                                    Fn &&fn, std::size_t block_size = 128) {
  std::size_t num_blocks = (numWorkItems.size() + block_size - 1) / block_size;

  int32_t range_size = numWorkItems.size();

  auto event = q.parallel_for(
      sycl::nd_range<>(num_blocks * block_size, block_size), [=](auto nd_idx) {
        auto idx = nd_idx.get_global_id(0);
        if (idx < range_size) {
          fn(idx);
        }
      });
  return event;
}

template <typename Fn>
sycl::event parallel_for_64bit(sycl::queue &q, sycl::range<1> numWorkItems,
                               Fn &&fn) {
  std::size_t block_size = 128;
  std::size_t max_kernel_size = max_kernel_size_(block_size);

  std::vector<sycl::event> events;
  for (std::size_t base_idx = 0; base_idx < numWorkItems.size();
       base_idx += max_kernel_size) {
    std::size_t launch_size =
        std::min(numWorkItems.size() - base_idx, max_kernel_size);

    auto e = parallel_for_workaround(
        q, launch_size,
        [=](sycl::id<1> idx_) {
          sycl::id<1> idx(base_idx + idx_);
          fn(idx);
        },
        block_size);

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
  std::size_t block_size = 128;
  std::size_t max_kernel_size = max_kernel_size_();

  if (numWorkItems.size() < max_kernel_size) {
    return parallel_for_workaround(q, numWorkItems, std::forward<Fn>(fn),
                                   block_size);
  } else {
    return parallel_for_64bit(q, numWorkItems, std::forward<Fn>(fn));
  }
}

} // namespace __detail

} // namespace dr::shp
