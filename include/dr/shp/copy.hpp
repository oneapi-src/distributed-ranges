// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "device_ptr.hpp"
#include <dr/concepts/concepts.hpp>
#include <dr/details/segments_tools.hpp>
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
template <lib::remote_contiguous_range R, std::contiguous_iterator OutputIt>
  requires __detail::is_syclmemcopyable<rng::range_value_t<R>,
                                        std::iter_value_t<OutputIt>>
sycl::event copy_async(const device_policy &policy, R &&r, OutputIt d_first) {
  return sycl::queue(shp::context(), policy.get_devices()[r.rank()])
      .copy(__detail::get_local_pointer(rng::begin(r)),
            std::to_address(d_first), rng::size(r));
}

/// Copy
template <lib::remote_contiguous_range R, std::contiguous_iterator OutputIt>
  requires __detail::is_syclmemcopyable<rng::range_value_t<R>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(const device_policy &policy, R &&r, OutputIt d_first) {
  const auto range_size = rng::size(r);
  copy_async(policy, std::forward(r), d_first).wait();
  return d_first + range_size;
}

// Copy from contiguous range to device
template <rng::random_access_range SrcRangeT,
          lib::remote_contiguous_range DstRangeT>
  requires __detail::is_syclmemcopyable<rng::range_value_t<SrcRangeT>,
                                        rng::range_value_t<DstRangeT>>
sycl::event copy_async(const device_policy &policy, SrcRangeT &&src,
                       DstRangeT &&dst) {
  return sycl::queue(shp::context(), policy.get_devices()[dst.rank()])
      .copy(std::to_address(rng::begin(src)),
            __detail::get_local_pointer(rng::begin(dst)),
            std::min(rng::size(src), rng::size(dst)));
}

template <rng::random_access_range SrcRangeT,
          lib::remote_contiguous_range DstRangeT>
  requires __detail::is_syclmemcopyable<rng::range_value_t<SrcRangeT>,
                                        rng::range_value_t<DstRangeT>>
auto copy(const device_policy &policy, SrcRangeT &&src, DstRangeT &&dst) {
  const auto range_size = std::min(rng::size(src), rng::size(dst));
  copy_async(policy, std::forward(src), std::forward(dst)).wait();
  return rng::begin(dst) + range_size;
}

// Copy from device to device
template <typename T>
  requires(!std::is_const_v<T> && std::is_trivially_copyable_v<T>)
sycl::event
    copy_async(device_ptr<std::add_const_t<T>> first,
               device_ptr<std::add_const_t<T>> last, device_ptr<T> d_first) {
  sycl::queue q;
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

    events.emplace_back(shp::copy_async(
        shp::par_unseq, rng::make_subrange(first, local_last), segment));

    ++segment_iter;
    first = local_last;
  }

  auto root_event =
      sycl::queue().submit([&](auto &&h) { h.depends_on(events); });
  return root_event;
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

    events.emplace_back(shp::copy_async(shp::par_unseq, segment, d_first));

    rng::advance(d_first, size);
  }

  auto root_event =
      sycl::queue().submit([&](auto &&h) { h.depends_on(events); });
  return root_event;
}

template <lib::distributed_iterator InputIt, std::forward_iterator OutputIt>
  requires __detail::is_syclmemcopyable<std::iter_value_t<InputIt>,
                                        std::iter_value_t<OutputIt>>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
  copy_async(first, last, d_first).wait();
  return d_first + (last - first);
}

// TODO: move fill code to seperate file
// TODO: get rid of fill versions without rank, they don't work on PVC
// fill with value
template <std::contiguous_iterator Iter>
  requires(!std::is_const_v<std::iter_value_t<Iter>> &&
           std::is_trivially_copyable_v<std::iter_value_t<Iter>>)
sycl::event
    fill_async(Iter first, Iter last, const std::iter_value_t<Iter> &value) {
  sycl::queue q;
  return q.fill(std::to_address(first), value, last - first);
}

sycl::event fill_async(const device_policy &policy,
                       lib::remote_contiguous_range auto &&r,
                       const auto &value) {
  using RangeValT = std::remove_cvref_t<decltype(*__detail::get_local_pointer(
      rng::begin(r)))>;
  const RangeValT val = value;
  return sycl::queue(shp::context(), policy.get_devices()[r.rank()])
      .fill(__detail::get_local_pointer(rng::begin(r)), val, rng::size(r));
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
  sycl::queue q;
  return q.fill(first.get_raw_pointer(), value, last - first);
}

template <typename T>
  requires(!std::is_const_v<T>)
void fill(device_ptr<T> first, device_ptr<T> last, const T &value) {
  fill_async(first, last, value).wait();
}

} // namespace shp
