// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>

#include <oneapi/dpl/execution>

#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/vector.hpp>
#include <oneapi/dpl/async>
#include <oneapi/dpl/numeric>

#include <dr/concepts/concepts.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace {

// Precondition: std::distance(begin, end) >= 2
// Postcondition: inclusive scan performed on [begin, end), return future
template <typename ExecutionPolicy, std::forward_iterator Iter,
          std::forward_iterator OutIter>
auto inclusive_scan_no_init_async(ExecutionPolicy &&policy, Iter begin,
                                  Iter end, OutIter d_begin) {
  return oneapi::dpl::experimental::inclusive_scan_async(
      std::forward<ExecutionPolicy>(policy), begin, end, d_begin);
}

} // namespace

namespace shp {

template <typename ExecutionPolicy, lib::distributed_range R>
void inclusive_scan(ExecutionPolicy &&policy, R &&r) {
  namespace sycl = cl::sycl;

  using T = rng::range_value_t<R>;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {

    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();
    auto &&segments = lib::ranges::segments(r);

    std::vector<sycl::event> events;

    auto root = devices[0];
    shp::device_allocator<T> allocator(shp::context(), root);
    shp::vector<T, shp::device_allocator<T>> partial_sums(
        std::size_t(segments.size()), allocator);

    std::size_t segment_id = 0;
    for (auto &&segment : segments) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(segment), rng::end(segment));

      sycl::event event;

      if (dist >= 2) {
        event = inclusive_scan_no_init_async(local_policy, rng::begin(segment),
                                             rng::end(segment),
                                             rng::begin(segment));
      }

      auto dst_iter = lib::ranges::local(partial_sums).data() + segment_id;
      auto src_iter = rng::begin(segment);
      std::advance(src_iter, dist - 1);
      auto e = q.submit([&](auto &&h) {
        h.depends_on(event);
        h.single_task([=]() { *dst_iter = *src_iter; });
      });

      events.push_back(e);

      segment_id++;
    }

    for (auto &&e : events) {
      e.wait();
    }
    events.clear();

    sycl::queue q(shp::context(), root);
    oneapi::dpl::execution::device_policy local_policy(q);

    auto first = lib::ranges::local(partial_sums).data();
    auto last = first + rng::size(partial_sums);

    oneapi::dpl::experimental::inclusive_scan_async(local_policy, first, last,
                                                    first)
        .wait();

    std::size_t idx = 0;
    for (auto &&segment : segments) {
      auto device = devices[lib::ranges::rank(segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(segment), rng::end(segment));

      rng::range_value_t<R> sum = 0;

      if (idx > 0) {
        sum = partial_sums[idx - 1];
      }

      sycl::event e = oneapi::dpl::experimental::for_each_async(
          local_policy, rng::begin(segment), rng::end(segment),
          [=](auto &&x) { x += sum; });

      events.push_back(e);
      idx++;
    }

    for (auto &&e : events) {
      e.wait();
    }

  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, lib::distributed_range R,
          lib::distributed_range O>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o) {
  namespace sycl = cl::sycl;

  using T = rng::range_value_t<O>;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  auto zipped_view = shp::views::zip(r, o);
  auto zipped_segments = zipped_view.zipped_segments();

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {

    auto &&devices = std::forward<ExecutionPolicy>(policy).get_devices();

    std::vector<sycl::event> events;

    auto root = devices[0];
    shp::device_allocator<T> allocator(shp::context(), root);
    shp::vector<T, shp::device_allocator<T>> partial_sums(
        std::size_t(zipped_segments.size()), allocator);

    std::size_t segment_id = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto device = devices[lib::ranges::rank(in_segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(in_segment), rng::end(in_segment));

      sycl::event event;

      if (dist >= 2) {
        fmt::print("Running on GPU {}\n", lib::ranges::rank(in_segment));
        event = inclusive_scan_no_init_async(
            local_policy, rng::begin(in_segment), rng::end(in_segment),
            rng::begin(out_segment));
      }

      auto dst_iter = lib::ranges::local(partial_sums).data() + segment_id;
      auto src_iter = rng::begin(out_segment);
      std::advance(src_iter, dist - 1);
      auto e = q.submit([&](auto &&h) {
        h.depends_on(event);
        h.single_task([=]() { *dst_iter = *src_iter; });
      });

      events.push_back(e);

      segment_id++;
    }

    for (auto &&e : events) {
      e.wait();
    }
    events.clear();

    sycl::queue q(shp::context(), root);
    oneapi::dpl::execution::device_policy local_policy(q);

    auto first = lib::ranges::local(partial_sums).data();
    auto last = first + rng::size(partial_sums);

    oneapi::dpl::experimental::inclusive_scan_async(local_policy, first, last,
                                                    first)
        .wait();

    std::size_t idx = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;
      auto device = devices[lib::ranges::rank(out_segment)];

      sycl::queue q(shp::context(), device);
      oneapi::dpl::execution::device_policy local_policy(q);

      auto dist = std::distance(rng::begin(out_segment), rng::end(out_segment));

      T sum = 0;

      if (idx > 0) {
        sum = partial_sums[idx - 1];
      }

      sycl::event e = oneapi::dpl::experimental::for_each_async(
          local_policy, rng::begin(out_segment), rng::end(out_segment),
          [=](auto &&x) { x = std::plus<>()(x, sum); });

      events.push_back(e);
      idx++;
    }

    for (auto &&e : events) {
      e.wait();
    }

  } else {
    assert(false);
  }
}

} // namespace shp
