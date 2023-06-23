// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/async>
#endif

namespace dr::mhp {
namespace __detail {

template <class FunT>
inline Event for_each_local_async(auto &&first, auto &&last, FunT &&func) {
  Event foreach_event;
#ifdef SYCL_LANGUAGE_VERSION
  if (use_sycl()) {
    foreach_event = oneapi::dpl::experimental::for_each_async(
        dpl_policy(), first, last, std::forward<FunT>(func));
  } else
#endif
  {
    std::for_each(std::execution::par_unseq, first, last,
                  std::forward<FunT>(func));
  }
  return foreach_event;
}

template <class FunT>
inline Event inclusive_scan_local_async(auto &&first, auto &&last,
                                        auto &&dst_first, FunT &&func) {
  Event scan_event;
#ifdef SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {
    scan_event = oneapi::dpl::experimental::inclusive_scan_async(
        dpl_policy(), first, last, dst_first, func);
  } else {
    std::inclusive_scan(std::execution::par_unseq, first, last, dst_first,
                        func);
  }
#else
  std::inclusive_scan(std::execution::par_unseq, first, last, dst_first, func);
#endif
  return scan_event;
}

template <class T>
inline Event copy_value_after_event(const T *src, T *dst, Event prerequisite) {
  Event e;
#ifdef SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {
    e = mhp::sycl_queue().submit([&](auto &&h) {
      h.depends_on(prerequisite);
      h.single_task([=]() {
        T tmp = *src;
        *dst = tmp;
      });
    });
  } else
#endif
  {
    *dst = *src;
  }
  return e;
}

// move to sycl_support.hpp if need to reuse
inline void wait_for_events_and_clear(std::vector<Event> &events) {
#ifdef SYCL_LANGUAGE_VERSION
  sycl::event::wait(events);
#endif
  events.clear();
}

// inclusive scan algorithm:
// let's denote segments as: rank(dot)segment_number
// let's segments be [1.1, 1.2, 2.1, 2.1, 3.1, 3.2]
// let's sum(X) denote collection of partial sum of range X
// 1. (offloaded) for each local segment compute partial sum
//    [..., sum(2.1), sum(2.2), ...]
// 2. get max (that is the last) elements of each local sum and create a vector
//    of it
//    [max(sum(2.1), max(sum(2.2)]
// 3. (offloaded) compute partial sum of above vector
//    sum([max(sum(2.1), max(sum(2.2)])
// 4. send max (that is the last element) of above sum to root process
//    let's denote above max of process no N as max_N
// 5. root process collects all maxes in an collection [max_1, max_2, max_3]
// 6. computes partial sum of it sum([max_1, max_2, max_3]) including init value
//    of inclusive_scan algorithm. Let's denote this partial sums as a "big sum"
// 7. big sum is scattered to all processes
// 8. (offloaded) Nth process gets big_sum_N element and adds it to each element
//    of local partial sums computed in step 3
// 9. each local i-th segment is modified by adding to each element value of
//    i-th partial sum computed in previous step
template <dr::distributed_contiguous_range R, dr::distributed_iterator O,
          typename BinaryOp, typename U = rng::range_value_t<R>>
auto inclusive_scan_impl_(R &&r, O &&d_first, BinaryOp &&binary_op,
                          std::optional<U> init = {}) {
  assert(aligned(r, d_first));

  auto d_last = d_first;
  rng::advance(d_last, rng::size(r));

  // TODO: current alg assumes local segments are contiguous and located
  //  in rank order. This is OK as all data structures we have currently have
  //  just one segment per rank and in rank order.
  auto in_segments = local_segments(r);
  auto out_segments = local_segments(rng::subrange(d_first, d_last));

  // rng::size(in_segments) doesn't work and can't count non-empty segments
  std::size_t local_segments_count = 0;
  for (auto &&s [[maybe_unused]] : in_segments)
    if (!rng::empty(s))
      ++local_segments_count;

  std::vector<Event> events;

  using OVal = rng::iter_value_t<O>;
  // how to get allocator used with input and output range? I assume it a
  // default one probably we should forbid using non-default ones with
  // distributed_ranges containers (like distributed_vector)
  auto local_partial_sums =
      default_allocator<OVal>().allocate_unique(local_segments_count);
  OVal *local_partial_sums_iter = local_partial_sums.get();

  for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

    auto &&[in_segment, out_segment] = segs;

    auto dist = rng::distance(in_segment);
    if (dist == 0) {
      continue;
    }

    auto s_first = ranges::local(rng::begin(out_segment));

    // this is step 1
    auto one_segment_scan_event = inclusive_scan_local_async(
        dr::__detail::direct_iterator(ranges::local(rng::begin(in_segment))),
        dr::__detail::direct_iterator(ranges::local(rng::end(in_segment))),
        dr::__detail::direct_iterator(s_first), binary_op);

    rng::advance(s_first, dist - 1);

    // this is step 2
    events.push_back(copy_value_after_event(s_first, local_partial_sums_iter,
                                            one_segment_scan_event));
    ++local_partial_sums_iter;
  }

  wait_for_events_and_clear(events);

  const std::size_t local_partial_sums_count =
      rng::distance(local_partial_sums.get(), local_partial_sums_iter);

  auto local_partial_sums_scanned =
      default_allocator<OVal>().allocate_unique(local_partial_sums_count);

  // this is step 3
  if (local_partial_sums_count) {
    inclusive_scan_local_async(local_partial_sums.get(),
                               local_partial_sums.get() +
                                   local_partial_sums_count,
                               local_partial_sums_scanned.get(), binary_op)
        .wait();
  }

  dr::communicator &comm = default_comm();
  std::optional<OVal> local_partial_sum =
      local_partial_sums_count
          ? local_partial_sums_scanned.get()[local_partial_sums_count - 1]
          : std::optional<OVal>();

  // below vector is used on 0 rank only but who cares
  std::vector<std::optional<OVal>> partial_sums(comm.size()); // dr-style ignore

  // this is step 4 and 5
  comm.gather(local_partial_sum, std::span{partial_sums}, 0);

  // this is step 6
  if (comm.rank() == 0) {
    std::optional<OVal> next_v = init;
    rng::for_each(partial_sums, [&next_v, binary_op](std::optional<OVal> &v) {
      if (v.has_value()) {
        std::swap(v, next_v);
        if (v.has_value()) // mind v was a next line before
          next_v = binary_op(v.value(), next_v.value());
      }
    });
  }

  // this is step 7
  std::optional<OVal> sum_of_all_guys_before_my_rank;
  comm.scatter(std::span{partial_sums}, sum_of_all_guys_before_my_rank, 0);

  // this is step 8
  if (local_partial_sums_count && sum_of_all_guys_before_my_rank.has_value()) {
    const OVal sum_of_all_guys_before_my_rank_value =
        sum_of_all_guys_before_my_rank.value();
    for_each_local_async(
        local_partial_sums_scanned.get(),
        local_partial_sums_scanned.get() + local_partial_sums_count,
        [=](auto &&x) {
          x = binary_op(sum_of_all_guys_before_my_rank_value, x);
        })
        .wait();
  }

  // this is step 9
  std::size_t local_partial_sum_idx = 0;
  bool nonempty_out_segment_already_found = false;
  for (auto &&out_seg : out_segments) {
    if (rng::empty(out_seg)) {
      continue;
    }

    auto first = ranges::local(rng::begin(out_seg));
    auto last = ranges::local(rng::end(out_seg));

    if (nonempty_out_segment_already_found ||
        sum_of_all_guys_before_my_rank.has_value()) {
      const OVal sum_of_all_guys_before_my_segment =
          nonempty_out_segment_already_found
              ? local_partial_sums_scanned.get()[local_partial_sum_idx - 1]
              : sum_of_all_guys_before_my_rank.value();
      events.push_back(for_each_local_async(
          dr::__detail::direct_iterator(first),
          dr::__detail::direct_iterator(last), [=](auto &&x) {
            x = binary_op(sum_of_all_guys_before_my_segment, x);
          }));
    }
    ++local_partial_sum_idx;
    nonempty_out_segment_already_found = true;
  }

  wait_for_events_and_clear(events);

  return d_last;
}

} // namespace __detail

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp, typename T>
auto inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op, T init) {
  return __detail::inclusive_scan_impl_(
      std::forward<R>(r), rng::begin(std::forward<O>(o)),
      std::forward<BinaryOp>(binary_op), std::optional(init));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp>
auto inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op) {
  return __detail::inclusive_scan_impl_(std::forward<R>(r),
                                        rng::begin(std::forward<O>(o)),
                                        std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O>
auto inclusive_scan(R &&r, O &&o) {
  return dr::mhp::inclusive_scan(std::forward<R>(r), std::forward<O>(o),
                                 std::plus<>());
}

// Distributed iterator versions

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename BinaryOp, typename T>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
                          BinaryOp &&binary_op, T init) {

  return dr::mhp::inclusive_scan(rng::subrange(first, last), d_first,
                                 std::forward<BinaryOp>(binary_op), init);
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename BinaryOp>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
                          BinaryOp &&binary_op) {

  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  dr::mhp::inclusive_scan(rng::subrange(first, last),
                          rng::subrange(d_first, d_last),
                          std::forward<BinaryOp>(binary_op));

  return d_last;
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first) {
  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  dr::mhp::inclusive_scan(rng::subrange(first, last),
                          rng::subrange(d_first, d_last));

  return d_last;
}

} // namespace dr::mhp
