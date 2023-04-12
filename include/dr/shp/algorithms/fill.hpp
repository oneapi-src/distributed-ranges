// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/segments_tools.hpp>
#include <dr/shp/device_ptr.hpp>
#include <dr/shp/util.hpp>

namespace shp {

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>> &&
           std::is_trivially_copyable_v<std::iter_value_t<Iter>>)
sycl::event fill_async(Iter first, Iter last,
                       const std::iter_value_t<Iter> &value) {
  auto &&q = shp::__detail::default_queue();
  std::iter_value_t<Iter> *arr = std::to_address(first);
  return q.parallel_for(sycl::range<>(last - first),
                        [arr, value](auto idx) { arr[idx] = value; });
  // not using q.fill because of CMPLRLLVM-46438
  // return q.fill(std::to_address(first), value, last - first);
}

template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>>)
void fill(Iter first, Iter last, const std::iter_value_t<Iter> &value) {
  fill_async(first, last, value).wait();
}

template <typename T>
  requires(!std::is_const_v<T>)
sycl::event fill_async(device_ptr<T> first, device_ptr<T> last,
                       const T &value) {
  auto &&q = shp::__detail::default_queue();
  auto *arr = first.get_raw_pointer();
  return q.parallel_for(sycl::range<>(last - first),
                        [arr, value](auto idx) { arr[idx] = value; });
  // not using q.fill because of CMPLRLLVM-46438
  // return q.fill(first.get_raw_pointer(), value, last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
void fill(device_ptr<T> first, device_ptr<T> last, const T &value) {
  fill_async(first, last, value).wait();
}

template <typename T, dr::remote_contiguous_range R>
sycl::event fill_async(R &&r, const T &value) {
  auto &&q = __detail::queue(dr::ranges::rank(r));
  auto *arr = std::to_address(rng::begin(dr::ranges::local(r)));
  return q.parallel_for(sycl::range<>(rng::distance(r)),
                        [arr, value](auto idx) { arr[idx] = value; });
  // not using q.fill because of CMPLRLLVM-46438
  // return q.fill(std::to_address(rng::begin(dr::ranges::local(r))), value,
  //               rng::distance(r));
}

template <typename T, dr::remote_contiguous_range R>
auto fill(R &&r, const T &value) {
  fill_async(r, value).wait();
  return rng::end(r);
}

template <typename T, dr::distributed_contiguous_range R>
sycl::event fill_async(R &&r, const T &value) {
  std::vector<sycl::event> events;

  for (auto &&segment : dr::ranges::segments(r)) {
    auto e = shp::fill_async(segment, value);
    events.push_back(e);
  }

  return shp::__detail::combine_events(events);
}

template <typename T, dr::distributed_contiguous_range R>
auto fill(R &&r, const T &value) {
  fill_async(r, value).wait();
  return rng::end(r);
}

template <typename T, dr::distributed_iterator Iter>
auto fill(Iter first, Iter last, const T &value) {
  fill_async(rng::subrange(first, last), value).wait();
  return last;
}

} // namespace shp
