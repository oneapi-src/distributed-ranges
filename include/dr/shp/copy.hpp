// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "device_ptr.hpp"
#include <CL/sycl.hpp>
#include <dr/concepts/concepts.hpp>
#include <dr/details/segments_tools.hpp>
#include <memory>
#include <type_traits>

namespace shp {

// TODO: in case the destination area is too small segfault may occur
// - add some error handling
template <std::contiguous_iterator Iter, typename T>
  requires(std::is_same_v<std::remove_const_t<std::iter_value_t<Iter>>, T> &&
           !std::is_const_v<T>)
cl::sycl::event copy_async(Iter first, Iter last, device_ptr<T> d_first) {
  cl::sycl::queue q;
  return q.memcpy(d_first.get_raw_pointer(), std::to_address(first),
                  sizeof(T) * (last - first));
}

template <std::contiguous_iterator Iter, typename T>
  requires(std::is_same_v<std::remove_const_t<std::iter_value_t<Iter>>, T> &&
           !std::is_const_v<T>)
device_ptr<T> copy(Iter first, Iter last, device_ptr<T> d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// TODO: in case the destination area is too small segfault may occur
// - add some error handling
template <typename T, std::contiguous_iterator Iter>
  requires(std::is_same_v<std::iter_value_t<Iter>, std::remove_const_t<T>>)
cl::sycl::event
    copy_async(device_ptr<T> first, device_ptr<T> last, Iter d_first) {
  cl::sycl::queue q;
  return q.memcpy(std::to_address(d_first), first.get_raw_pointer(),
                  sizeof(T) * (last - first));
}

template <std::contiguous_iterator Iter, typename T>
  requires(std::is_same_v<std::remove_const_t<std::iter_value_t<Iter>>, T> &&
           !std::is_const_v<T>)
Iter copy(device_ptr<const T> first, device_ptr<const T> last, Iter d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
cl::sycl::event
    fill_async(device_ptr<T> first, device_ptr<T> last, const T &value) {
  cl::sycl::queue q;
  return q.fill(first.get_raw_pointer(), value, last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
void fill(device_ptr<T> first, device_ptr<T> last, const T &value) {
  cl::sycl::queue q;
  q.fill(first.get_raw_pointer(), value, last - first).wait();
}

// Copy from local range to distributed range
template <std::forward_iterator InputIt, lib::distributed_iterator OutputIt>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  auto segments = lib::ranges::segments(d_first);

  std::size_t segment_id = 0;
  std::size_t local_id = 0;

  std::vector<cl::sycl::event> events;

  std::size_t total_copied = 0;

  while (first != last) {
    auto &&segment = segments[segment_id];
    std::size_t n_in_segment = segment.size() - local_id;

    std::size_t n_to_copy =
        std::min<size_t>(n_in_segment, std::distance(first, last));

    auto local_last = first;
    std::advance(local_last, n_to_copy);

    auto remote_iter = segment.begin();
    std::advance(remote_iter, local_id);

    auto event = shp::copy_async(first, local_last, remote_iter);

    events.push_back(event);

    local_id = 0;
    segment_id++;
    std::advance(first, n_to_copy);
    total_copied += n_to_copy;
  }

  for (auto &&event : events) {
    event.wait();
  }

  auto rv = d_first;
  std::advance(rv, total_copied);
  return rv;
}

template <std::forward_iterator InputIt, lib::distributed_iterator OutputIt>
cl::sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto segments = lib::ranges::segments(d_first);

  std::size_t segment_id = lib::ranges::segment_index(d_first);
  std::size_t local_id = lib::ranges::local_index(d_first);

  std::vector<cl::sycl::event> events;

  std::size_t total_copied = 0;

  while (first != last) {
    auto &&segment = segments[segment_id];
    std::size_t n_in_segment = segment.size() - local_id;

    std::size_t n_to_copy =
        std::min<size_t>(n_in_segment, std::distance(first, last));

    auto local_last = first;
    std::advance(local_last, n_to_copy);

    auto remote_iter = segment.begin();
    std::advance(remote_iter, local_id);

    auto event = shp::copy_async(first, local_last, remote_iter);

    events.push_back(event);

    local_id = 0;
    segment_id++;
    std::advance(first, n_to_copy);
    total_copied += n_to_copy;
  }

  sycl::queue q;

  auto root_event = q.submit([=](auto &&h) { h.depends_on(events); });

  return root_event;
}

// Copy from distributed range to local range
template <lib::distributed_iterator InputIt, std::forward_iterator OutputIt>
cl::sycl::event copy_async(InputIt first, InputIt last, OutputIt d_first) {
  auto size = std::distance(first, last);
  auto segments =
      lib::internal::take_segments(lib::ranges::segments(first), size);

  std::vector<cl::sycl::event> events;

  for (auto &&segment : segments) {
    auto event =
        shp::copy_async(rng::begin(segment), rng::end(segment), d_first);

    events.push_back(event);

    std::advance(d_first, rng::size(segment));
  }

  sycl::queue q;

  auto root_event = q.submit([=](auto &&h) { h.depends_on(events); });

  return root_event;
}

template <lib::distributed_iterator InputIt, std::forward_iterator OutputIt>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  std::advance(d_first, last - first);
  return d_first;
}

} // namespace shp
