// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <limits>

#ifdef SYCL_LANGUAGE_VERSION

#include <sycl/sycl.hpp>

#define Dim 1

namespace dr::__detail {

inline const std::array dim0 = {0, 0, 0}, dim1 = {128, 0, 0},
                        dim2 = {16, 16, 0}, dim3 = {8, 8, 8};
inline const std::array workgroup_shapes = {dim0, dim1, dim2, dim3};

//
// return true if the device can be partitoned by affinity domain
//
inline auto partitionable(sycl::device device) {
  // Earlier commits used the query API, but they return true even
  // though a partition will fail:  Intel MPI mpirun with multiple
  // processes.
  try {
    device.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::numa);
  } catch (sycl::exception const &e) {
    if (e.code() == sycl::errc::invalid ||
        e.code() == sycl::errc::feature_not_supported) {
      return false;
    } else {
      throw;
    }
  }

  return true;
}

// With the ND-range workaround, the maximum kernel size is
// `std::numeric_limits<std::int32_t>::max()` rounded down to
// the nearest multiple of the block size.
inline std::size_t max_kernel_size_(std::size_t block_size) {
  std::size_t max_kernel_size = std::numeric_limits<std::int32_t>::max();
  return (max_kernel_size / block_size) * block_size;
}

// This is a workaround to avoid performance degradation
// in DPC++ for odd range sizes.
template <typename Fn>
sycl::event parallel_for_workaround(sycl::queue &q,
                                    sycl::range<Dim> numWorkItems, Fn &&fn) {
  auto block_size = workgroup_shapes[Dim][0];
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
sycl::event parallel_for_64bit(sycl::queue &q, sycl::range<Dim> numWorkItems,
                               Fn &&fn) {
  auto block_size = workgroup_shapes[Dim][0];
  std::size_t max_kernel_size = max_kernel_size_(block_size);

  std::vector<sycl::event> events;
  for (std::size_t base_idx = 0; base_idx < numWorkItems.size();
       base_idx += max_kernel_size) {
    std::size_t launch_size =
        std::min(numWorkItems.size() - base_idx, max_kernel_size);

    auto e = parallel_for_workaround(q, sycl::range<1>(launch_size),
                                     [=](sycl::id<1> idx_) {
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
sycl::event parallel_for(sycl::queue &q, sycl::range<Dim> range, Fn &&fn) {
  const auto workgroup_shape = workgroup_shapes[Dim];

  bool overflow = false;
  for (std::size_t i = 0; i < Dim; i++) {
    if (range[i] >= max_kernel_size_(workgroup_shape[i])) {
      overflow = true;
    }
  }
  if (overflow) {
    return parallel_for_64bit(q, range, std::forward<Fn>(fn));
  } else {
    return parallel_for_workaround(q, range, std::forward<Fn>(fn));
  }
}

} // namespace dr::__detail

#endif // SYCL_LANGUAGE_VERSION
