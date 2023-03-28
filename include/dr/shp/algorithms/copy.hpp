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
namespace __detail {

template <typename Src, typename Dest>
concept is_syclmemcopyable = std::is_same_v<std::remove_const_t<Src>, Dest> &&
                             std::is_trivially_copyable_v<Dest>;
} // namespace __detail

// TODO: move copy file into algorithms directory
// Copy between contiguous ranges
template <std::contiguous_iterator InputIt, std::contiguous_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto q = shp::__detail::default_queue();
  return q.memcpy(std::to_address(d_first), std::to_address(first),
                  sizeof(std::iter_value_t<InputIt>) * (last - first));
}

/// Copy
template <std::contiguous_iterator InputIt, std::contiguous_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from contiguous range to device
template <std::contiguous_iterator Iter, typename T>
  requires __detail::is_syclmemcopyable<std::iter_value_t<Iter>, T>
sycl::event copy_async(Iter first, Iter last, device_ptr<T> d_first) {
  auto q = shp::__detail::default_queue();
  return q.memcpy(d_first.get_raw_pointer(), std::to_address(first),
                  sizeof(T) * (last - first));
}

template <std::contiguous_iterator Iter, typename T>
  requires __detail::is_syclmemcopyable<std::iter_value_t<Iter>, T>
device_ptr<T> copy(Iter first, Iter last, device_ptr<T> d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from device to contiguous range
template <typename T, std::contiguous_iterator Iter>
  requires __detail::is_syclmemcopyable<T, std::iter_value_t<Iter>>
sycl::event copy_async(device_ptr<T> first, device_ptr<T> last, Iter d_first) {
  auto q = shp::__detail::default_queue();
  return q.memcpy(std::to_address(d_first), first.get_raw_pointer(),
                  sizeof(T) * (last - first));
}

template <typename T, std::contiguous_iterator Iter>
  requires __detail::is_syclmemcopyable<T, std::iter_value_t<Iter>>
Iter copy(device_ptr<T> first, device_ptr<T> last, Iter d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from device to device
template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T>)
sycl::event
    copy_async(device_ptr<std::add_const_t<T>> first,
               device_ptr<std::add_const_t<T>> last, device_ptr<T> d_first) {
  auto q = shp::__detail::default_queue();
  return q.memcpy(d_first.get_raw_pointer(), first.get_raw_pointer(),
                  sizeof(T) * (last - first));
}

template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T>)
device_ptr<T> copy(device_ptr<std::add_const_t<T>> first,
                   device_ptr<std::add_const_t<T>> last,
                   device_ptr<T> d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from local range to distributed range
template <std::forward_iterator InputIt, lib::distributed_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto &&segments = lib::ranges::segments(d_first);
  auto segment_iter = rng::begin(segments);

  std::vector<sycl::event> events;

  while (first != last) {
    auto &&segment = *segment_iter;
    auto size = rng::distance(segment);

    std::size_t n_to_copy = std::min<size_t>(size, rng::distance(first, last));

    auto local_last = first;
    rng::advance(local_last, n_to_copy);

    events.emplace_back(
        shp::copy_async(first, local_last, rng::begin(segment)));

    ++segment_iter;
    rng::advance(first, n_to_copy);
  }

  return shp::__detail::combine_events(events);
}

template <std::contiguous_iterator InputIt, lib::distributed_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// Copy from distributed range to local range
template <lib::distributed_iterator InputIt, std::forward_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto dist = rng::distance(first, last);
  auto segments =
      lib::internal::take_segments(lib::ranges::segments(first), dist);

  std::vector<sycl::event> events;

  for (auto &&segment : segments) {
    auto size = rng::distance(segment);

    events.emplace_back(
        shp::copy_async(rng::begin(segment), rng::end(segment), d_first));

    rng::advance(d_first, size);
  }

  return shp::__detail::combine_events(events);
}

template <lib::distributed_iterator InputIt, std::forward_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// iota
template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>> &&
           std::is_trivially_copyable_v<std::iter_value_t<Iter>>)
sycl::event
    iota_async(Iter first, Iter last, const std::iter_value_t<Iter> value) {
  auto ptr = std::to_address(first);
  return sycl::queue().parallel_for(sycl::range<1>(last - first),
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
  return sycl::queue().parallel_for(sycl::range<1>(last - first),
                                    [=](auto id) { ptr[id] = value + id; });
}

template <typename T>
  requires(!std::is_const_v<T>)
void iota(device_ptr<T> first, device_ptr<T> last, const T value) {
  iota_async(first, last, value).wait();
}

template <typename T, lib::remote_contiguous_range R>
sycl::event iota_async(R &&r, const T value) {
  auto ptr = std::to_address(rng::begin(lib::ranges::local(r)));
  return sycl::queue().parallel_for(sycl::range<1>(rng::distance(r)),
                                    [=](auto id) { ptr[id] = value + id; });
}

template <typename T, lib::remote_contiguous_range R>
auto iota(R &&r, const T value) {
  iota_async(r, value).wait();
  return rng::end(r);
}

template <typename T, lib::distributed_contiguous_range R>
sycl::event iota_async(R &&r, const T value) {
  std::vector<sycl::event> events;

  for (auto &&segment : lib::ranges::segments(r)) {
    auto e = shp::iota_async(segment, value);
    events.push_back(e);
  }

  return shp::__detail::combine_events(events);
}

template <typename T, lib::distributed_contiguous_range R>
auto iota(R &&r, const T value) {
  iota_async(r, value).wait();
  return rng::end(r);
}

} // namespace shp
