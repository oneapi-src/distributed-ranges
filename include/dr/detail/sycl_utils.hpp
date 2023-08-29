// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <limits>

#include <dr/detail/utils.hpp>

#ifdef SYCL_LANGUAGE_VERSION

#include <sycl/sycl.hpp>

namespace dr::__detail {

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

// Convert a global range to a nd_range using generic block size
template <int Dim> auto nd_range(sycl::range<Dim> global_size) {
  if constexpr (Dim == 1) {
    return sycl::nd_range<Dim>(global_size, sycl::range<Dim>(128));
  } else if constexpr (Dim == 2) {
    return sycl::nd_range<Dim>(global_size, sycl::range<Dim>(16, 16));
  } else if constexpr (Dim == 3) {
    return sycl::nd_range<Dim>(global_size, sycl::range<Dim>(8, 8, 8));
  } else {
    assert(false);
    return sycl::range<>(0);
  }
}

template <typename Fn>
sycl::event parallel_for_nd(sycl::queue &q, sycl::range<1> global_size,
                            Fn &&fn) {
  return q.parallel_for(nd_range(global_size),
                        [=](auto nd_idx) { fn(nd_idx.get_global_id(0)); });
}

template <typename Fn>
sycl::event parallel_for_nd(sycl::queue &q, sycl::range<2> global_size,
                            Fn &&fn) {
  return q.parallel_for(nd_range(global_size), [=](auto nd_idx) {
    fn(std::array{nd_idx.get_global_id(0), nd_idx.get_global_id(1)});
  });
}

template <typename Fn>
sycl::event parallel_for_nd(sycl::queue &q, sycl::range<3> global_size,
                            Fn &&fn) {
  return q.parallel_for(nd_range(global_size), [=](auto nd_idx) {
    fn(std::array{nd_idx.get_global_id(0), nd_idx.get_global_id(1),
                  nd_idx.get_global_id(2)});
  });
}

auto combine_events(sycl::queue &q, const auto &events) {
  return q.submit([&](auto &&h) {
    h.depends_on(events);
    // Empty host task necessary due to [CMPLRLLVM-46542]
    h.host_task([] {});
  });
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<1> global_size, Fn &&fn) {
  std::vector<sycl::event> events;

  // Chunks are 32 bits
  for (std::size_t remainder = global_size[0]; remainder != 0;) {
    std::size_t chunk = std::min(
        remainder, std::size_t(std::numeric_limits<std::int32_t>::max()));
    events.push_back(parallel_for_nd(q, sycl::range<>(chunk), fn));
    remainder -= chunk;
  }

  return combine_events(q, events);
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<2> global_size, Fn &&fn) {
  auto max = std::numeric_limits<std::int32_t>::max();
  assert(global_size[0] < max && global_size[1] < max);
  return parallel_for_nd(q, global_size, fn);
}

template <typename Fn>
sycl::event parallel_for(sycl::queue &q, sycl::range<3> global_size, Fn &&fn) {
  auto max = std::numeric_limits<std::int32_t>::max();
  assert(global_size[0] < max && global_size[1] < max && global_size[2] < max);
  return parallel_for_nd(q, global_size, fn);
}

} // namespace dr::__detail

#endif // SYCL_LANGUAGE_VERSION
