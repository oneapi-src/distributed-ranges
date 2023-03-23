// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/concepts/concepts.hpp>
#include <dr/details/segments_tools.hpp>
#include <dr/shp/device_ptr.hpp>
#include <memory>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace shp {

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>> &&
           std::is_trivially_copyable_v<std::iter_value_t<Iter>>)
sycl::event
    fill_async(Iter first, Iter last, const std::iter_value_t<Iter> &value) {
  auto q = shp::__detail::default_queue();
  return q.fill(std::to_address(first), value, last - first);
}

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>>)
void fill(Iter first, Iter last, const std::iter_value_t<Iter> &value) {
  fill_async(first, last, value).wait();
}

template <typename T>
  requires(!std::is_const_v<T>)
sycl::event
    fill_async(device_ptr<T> first, device_ptr<T> last, const T &value) {
  auto q = shp::__detail::default_queue();
  return q.fill(first.get_raw_pointer(), value, last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
void fill(device_ptr<T> first, device_ptr<T> last, const T &value) {
  fill_async(first, last, value).wait();
}

template <typename T, lib::remote_contiguous_range R>
sycl::event fill_async(R &&r, const T &value) {
  sycl::queue q(shp::context(), shp::devices()[lib::ranges::rank(r)]);
  return q.fill(std::to_address(rng::begin(lib::ranges::local(r))), value,
                rng::distance(r));
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
