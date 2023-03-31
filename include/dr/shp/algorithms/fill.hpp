// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/details/segments_tools.hpp>
#include <dr/shp/device_ptr.hpp>
#include <dr/shp/util.hpp>
#include <memory>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace shp {

template <typename T, lib::remote_contiguous_range R>
sycl::event fill_async(R &&r, const T &value) {
  // fill can not be directed to node not owning memory being filled
  auto q = __detail::queue_for_rank(lib::ranges::rank(r));
  T *arr = std::to_address(rng::begin(lib::ranges::local(r)));
  return q.parallel_for(sycl::range<1>(rng::distance(r)),
                        [arr, value](sycl::id<1> idx) { arr[idx] = value; });
}

template <typename T, lib::remote_contiguous_range R>
auto fill(R &&r, const T &value) {
  fill_async(r, value).wait();
  return rng::end(r);
}

template <typename T, lib::distributed_contiguous_range R>
sycl::event fill_async(R &&r, const T &value) {
  std::vector<sycl::event> events;

  for (auto &&segment : lib::ranges::segments(r)) {
    auto e = shp::fill_async(segment, value);
    events.push_back(e);
  }

  return shp::__detail::combine_events(events);
}

template <typename T, lib::distributed_contiguous_range R>
auto fill(R &&r, const T &value) {
  fill_async(r, value).wait();
  return rng::end(r);
}

} // namespace shp
