// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/shp/device_ptr.hpp>
#include <dr/shp/util.hpp>

namespace shp {

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>> &&
           std::is_trivially_copyable_v<std::iter_value_t<Iter>>)
sycl::event iota_async(Iter first, Iter last,
                       const std::iter_value_t<Iter> value) {
  auto ptr = std::to_address(first);
  return sycl::queue().parallel_for(sycl::range<>(last - first),
                                    [=](auto id) { ptr[id] = value + id; });
}

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>>)
void iota(Iter first, Iter last, const std::iter_value_t<Iter> value) {
  iota_async(first, last, value).wait();
}

template <typename T>
  requires(!std::is_const_v<T>)
sycl::event iota_async(device_ptr<T> first, device_ptr<T> last, const T value) {
  auto ptr = first.get_raw_pointer();
  return sycl::queue().parallel_for(sycl::range<>(last - first),
                                    [=](auto id) { ptr[id] = value + id; });
}

template <typename T>
  requires(!std::is_const_v<T>)
void iota(device_ptr<T> first, device_ptr<T> last, const T value) {
  iota_async(first, last, value).wait();
}

template <lib::remote_contiguous_range R, typename T>
sycl::event iota_async(R &&r, const T value) {
  auto ptr = std::to_address(rng::begin(lib::ranges::local(r)));
  return __detail::queue_for_rank(lib::ranges::rank(r))
      .parallel_for(sycl::range<>(rng::distance(r)),
                    [=](auto id) { ptr[id] = value + id; });
}

template <lib::remote_contiguous_range R, typename T>
void iota(R &&r, const T value) {
  iota_async(r, value).wait();
}

template <lib::distributed_contiguous_range R, typename T>
sycl::event iota_async(R &&r, const T value) {
  std::vector<sycl::event> events;

  auto init_value = value;
  for (auto &&segment : lib::ranges::segments(r)) {
    auto e = shp::iota_async(segment, init_value);
    events.push_back(e);
    init_value = init_value + segment.size();
  }

  return shp::__detail::combine_events(events);
}

template <lib::distributed_contiguous_range R, typename T>
void iota(R &&r, const T value) {
  iota_async(r, value).wait();
}

template <lib::distributed_iterator DistIt, typename T>
void iota(DistIt first, DistIt last, const T value) {
  iota(rng::subrange(first, last), value);
}

} // namespace shp
