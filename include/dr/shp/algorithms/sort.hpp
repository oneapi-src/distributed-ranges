// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/execution>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/shp/init.hpp>
#include <sycl/sycl.hpp>

namespace dr {

namespace shp {

namespace __detail {

template <typename LocalPolicy, typename InputIt, typename Compare>
sycl::event sort_async(LocalPolicy &&policy, InputIt first, InputIt last,
                       Compare &&comp) {
  if (rng::distance(first, last) >= 2) {
    dr::__detail::direct_iterator d_first(first);
    dr::__detail::direct_iterator d_last(last);
    return oneapi::dpl::experimental::sort_async(
        std::forward<LocalPolicy>(policy), d_first, d_last,
        std::forward<Compare>(comp));
  } else {
    return sycl::event{};
  }
}

} // namespace __detail

template <dr::distributed_range R, typename Compare = std::less<>>
void sort(R &&r, Compare comp = Compare()) {
  auto &&segments = dr::ranges::segments(r);

  if (rng::size(segments) == 0) {
    return;
  } else if (rng::size(segments) == 1) {
    auto &&segment = *rng::begin(segments);
    auto &&local_policy =
        dr::shp::__detail::dpl_policy(dr::ranges::rank(segment));
    auto &&local_segment = dr::shp::__detail::local(segment);

    __detail::sort_async(local_policy, rng::begin(segment), rng::end(segment),
                         comp)
        .wait();
    return;
  }

  using T = rng::range_value_t<R>;
  std::vector<sycl::event> events;

  std::size_t n_segments = std::size_t(rng::size(segments));
  std::size_t n_splitters = n_segments - 1;

  // Sort each local segment, then compute medians.
  // Each segment has `n_splitters` medians,
  // so `n_segments * n_splitters` medians total.

  T *medians = sycl::malloc_shared<T>(n_segments * n_splitters,
                                      shp::devices()[0], shp::context());

  std::size_t segment_id = 0;

  for (auto &&segment : segments) {
    auto &&q = dr::shp::__detail::queue(dr::ranges::rank(segment));
    auto &&local_policy =
        dr::shp::__detail::dpl_policy(dr::ranges::rank(segment));

    auto &&local_segment = dr::shp::__detail::local(segment);

    auto s = __detail::sort_async(local_policy, rng::begin(local_segment),
                                  rng::end(local_segment), comp);

    double step_size = static_cast<double>(rng::size(segment)) / n_segments;

    auto local_begin = rng::begin(local_segment);

    auto e = q.submit([&](auto &&h) {
      h.depends_on(s);

      h.parallel_for(n_splitters, [=](auto i) {
        medians[n_splitters * segment_id + i] =
            local_begin[std::size_t(step_size * (i + 1) + 0.5)];
      });
    });

    events.push_back(e);
    ++segment_id;
  }

  dr::shp::__detail::wait(events);
  events.clear();

  // Compute global medians by sorting medians and
  // computing `n_splitters` medians from the medians.
  auto &&local_policy = dr::shp::__detail::dpl_policy(0);
  __detail::sort_async(local_policy, medians,
                       medians + n_segments * n_splitters, comp)
      .wait();

  double step_size = static_cast<double>(n_segments * n_splitters) / n_segments;

  // - Collect median of medians to get final splitters.
  // - Write splitters to [0, n_splitters) in `medians`
  for (std::size_t i = 0; i < n_splitters; i++) {
    medians[i] = medians[std::size_t(step_size * (i + 1) + 0.5)];
  }

  /*
    fmt::print("Medians: [");
    for (std::size_t i = 0; i < n_splitters; i++) {
      fmt::print("{}", medians[i]);

      if (i +1 < n_splitters) {
        fmt::print(", ");
      }
    }
    fmt::print("]\n");
    */

  std::vector<std::size_t *> splitter_indices;
  std::vector<std::size_t> sorted_seg_sizes(n_splitters + 1);
  std::vector<std::vector<std::size_t>> push_positions(n_segments);

  // Compute how many elements will be sent to each of the new "sorted
  // segments". Simultaneously compute the offsets `push_positions` where each
  // segments' corresponding elements will be pushed.

  segment_id = 0;
  for (auto &&segment : segments) {
    auto &&q = dr::shp::__detail::queue(dr::ranges::rank(segment));
    auto &&local_policy =
        dr::shp::__detail::dpl_policy(dr::ranges::rank(segment));

    auto &&local_segment = dr::shp::__detail::local(segment);
    dr::__detail::direct_iterator local_begin(rng::begin(local_segment));
    dr::__detail::direct_iterator local_end(rng::end(local_segment));

    // fmt::print("Segment {}: {}\n", dr::ranges::rank(segment), local_segment);

    std::size_t *splitter_i = sycl::malloc_shared<std::size_t>(n_splitters, q);
    splitter_indices.push_back(splitter_i);

    oneapi::dpl::lower_bound(local_policy, local_begin, local_end, medians,
                             medians + n_splitters, splitter_i, comp);

    // fmt::print("splitter indices: {}\n", rng::subrange(splitter_i, splitter_i
    // + n_splitters));

    auto p_first = rng::begin(local_segment);
    auto p_last = p_first;
    for (std::size_t i = 0; i < n_splitters; i++) {
      p_last = rng::begin(local_segment) + splitter_i[i];

      // fmt::print("Will need to send {}\n", rng::subrange(p_first, p_last));

      std::size_t n_elements = rng::distance(p_first, p_last);
      std::size_t pos =
          std::atomic_ref(sorted_seg_sizes[i]).fetch_add(n_elements);

      push_positions[segment_id].push_back(pos);

      p_first = p_last;
    }

    // fmt::print("Will need to send {}\n", rng::subrange(p_first,
    // rng::end(local_segment)));

    std::size_t n_elements = rng::distance(p_first, rng::end(local_segment));
    std::size_t pos =
        std::atomic_ref(sorted_seg_sizes.back()).fetch_add(n_elements);

    push_positions[segment_id].push_back(pos);

    ++segment_id;
  }

  // fmt::print("sorted segment sizes: {}\n", sorted_seg_sizes);

  // Allocate new "sorted segments"
  std::vector<T *> sorted_segments;

  segment_id = 0;
  for (auto &&segment : segments) {
    auto &&q = dr::shp::__detail::queue(dr::ranges::rank(segment));

    T *buffer = sycl::malloc_device<T>(sorted_seg_sizes[segment_id], q);
    sorted_segments.push_back(buffer);

    ++segment_id;
  }

  // Copy corresponding elements to each "sorted segment"
  segment_id = 0;
  for (auto &&segment : segments) {
    auto &&q = dr::shp::__detail::queue(dr::ranges::rank(segment));
    auto &&local_policy =
        dr::shp::__detail::dpl_policy(dr::ranges::rank(segment));

    auto &&local_segment = dr::shp::__detail::local(segment);

    std::size_t *splitter_i = splitter_indices[segment_id];

    auto p_first = rng::begin(local_segment);
    auto p_last = p_first;
    for (std::size_t i = 0; i < n_splitters; i++) {
      p_last = rng::begin(local_segment) + splitter_i[i];

      // fmt::print("Sending {}\n", rng::subrange(p_first, p_last));

      std::size_t n_elements = rng::distance(p_first, p_last);
      std::size_t pos = push_positions[segment_id][i];

      auto e = shp::copy_async(p_first, p_last, sorted_segments[i] + pos);
      events.push_back(e);

      p_first = p_last;
    }

    // fmt::print("Sending {}\n", rng::subrange(p_first,
    // rng::end(local_segment)));

    std::size_t n_elements = rng::distance(p_first, rng::end(local_segment));
    std::size_t pos = push_positions[segment_id].back();

    auto e = shp::copy_async(p_first, rng::end(local_segment),
                             sorted_segments.back() + pos);

    events.push_back(e);

    ++segment_id;
  }

  dr::shp::__detail::wait(events);
  events.clear();

  // Sort each of these new segments
  for (std::size_t i = 0; i < sorted_segments.size(); i++) {
    auto &&local_policy =
        dr::shp::__detail::dpl_policy(dr::ranges::rank(segments[i]));
    T *seg = sorted_segments[i];
    std::size_t n_elements = sorted_seg_sizes[i];

    auto e = __detail::sort_async(local_policy, seg, seg + n_elements, comp);

    events.push_back(e);
  }

  dr::shp::__detail::wait(events);
  events.clear();

  // Copy the results into the output.

  auto d_first = rng::begin(r);

  for (std::size_t i = 0; i < sorted_segments.size(); i++) {
    T *seg = sorted_segments[i];
    std::size_t n_elements = sorted_seg_sizes[i];

    auto e = shp::copy_async(seg, seg + n_elements, d_first);

    events.push_back(e);

    rng::advance(d_first, n_elements);
  }

  dr::shp::__detail::wait(events);

  // Free temporary memory.

  for (auto &&sorted_seg : sorted_segments) {
    sycl::free(sorted_seg, shp::context());
  }

  for (auto &&splitter_i : splitter_indices) {
    sycl::free(splitter_i, shp::context());
  }

  sycl::free(medians, shp::context());
}

} // namespace shp

} // namespace dr
