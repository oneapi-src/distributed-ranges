// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/async>
#endif

namespace dr::mhp {
namespace __detail {

template <typename R> bool is_first_nonempty_segment_local(R &&dr) {
  for (auto &&s : dr::ranges::segments(std::forward<R>(dr))) {
    if (!rng::empty(s)) {
      return dr::ranges::rank(s) == default_comm().rank();
    }
  }
  return false;
}

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

template <class FunT, class... Init>
inline void inclusive_scan_local_on_cpu(auto &&first, auto &&last,
                                        auto &&dst_first, FunT &&func,
                                        Init &&...init) {
  std::inclusive_scan(std::execution::par_unseq, first, last, dst_first, func,
                      std::forward<Init>(init)...);
}

template <class FunT, class... Init>
inline Event inclusive_scan_local_async(auto &&first, auto &&last,
                                        auto &&dst_first, FunT &&func,
                                        Init &&...init) {
  Event scan_event;
#ifdef SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {
    scan_event = oneapi::dpl::experimental::inclusive_scan_async(
        dpl_policy(), first, last, dst_first, func,
        std::forward<Init>(init)...);
  } else {
    inclusive_scan_local_on_cpu(first, last, dst_first, func,
                                std::forward<Init>(init)...);
  }
#else
  inclusive_scan_local_on_cpu(first, last, dst_first, func,
                              std::forward<Init>(init)...);
#endif
  return scan_event;
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
//    first non-empty partial sum uses init value
// 2. get max (that is the last) elements of each local sum and create a vector
//    of it
//    [max(sum(2.1), max(sum(2.2)]
// 3. compute partial sum of above vector
//    sum([max(sum(2.1), max(sum(2.2)])
// 4. send max (that is the last element) of above sum to root process
//    let's denote above max of process no N as max_N
// 5. root process collects all maxes in an collection [max_1, max_2, max_3]
// 6. computes partial sum of it sum([max_1, max_2, max_3])
//    Let's denote this partial sums as a "big sum".
// 7. big sum is scattered to all processes
// 8. Nth process gets big_sum_N element and adds it to each element of local
//    partial sums computed in step 3
// 9. (offloaded) each local i-th segment is modified by adding to each element
//    value of i-th partial sum computed in previous step
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

  std::vector<Event> events;

  using OVal = rng::iter_value_t<O>;

  bool const first_nonempty_segment_is_local =
      is_first_nonempty_segment_local(r);
  bool include_init_into_sum =
      init.has_value() && first_nonempty_segment_is_local;

  // this is step 1
  for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

    auto &&[in_segment, out_segment] = segs;
    if (rng::empty(in_segment))
      continue;

    auto first =
        dr::__detail::direct_iterator(ranges::local(rng::begin(in_segment)));
    auto last =
        dr::__detail::direct_iterator(ranges::local(rng::end(in_segment)));
    auto s_first =
        dr::__detail::direct_iterator(ranges::local(rng::begin(out_segment)));

    events.push_back(
        include_init_into_sum
            ? inclusive_scan_local_async(first, last, s_first, binary_op,
                                         init.value())
            : inclusive_scan_local_async(first, last, s_first, binary_op));
    include_init_into_sum = false;
  }
  wait_for_events_and_clear(events);

  // this is step 2
  std::vector<OVal> local_partial_sums;
  for (auto &&out_segment : out_segments) {
    auto dist = rng::distance(out_segment);
    if (dist == 0)
      continue;

    auto s_first = ranges::local(rng::begin(out_segment));
    rng::advance(s_first, dist - 1);
    local_partial_sums.push_back(*s_first);
  }

  // this is step 3
  std::vector<OVal> local_partial_sums_scanned(rng::size(local_partial_sums));
  inclusive_scan_local_on_cpu(local_partial_sums.begin(),
                              local_partial_sums.end(),
                              local_partial_sums_scanned.begin(), binary_op);

  dr::communicator &comm = default_comm();
  std::optional<OVal> local_partial_sum =
      rng::empty(local_partial_sums_scanned)
          ? std::optional<OVal>()
          : local_partial_sums_scanned.back();

  // below vector is used on 0 rank only but who cares
  std::vector<std::optional<OVal>> partial_sums(comm.size());

  // this is step 4 and 5
  comm.gather(local_partial_sum, std::span{partial_sums}, 0);

  // this is step 6
  std::vector<std::optional<OVal>> partial_sums_scanned(comm.size() + 1);
  if (comm.rank() == 0) {
    inclusive_scan_local_on_cpu(
        rng::begin(partial_sums), rng::end(partial_sums),
        rng::begin(partial_sums_scanned) + 1,
        [binary_op](std::optional<OVal> x, std::optional<OVal> y) {
          if (x.has_value() && y.has_value()) {
            return std::make_optional(binary_op(x.value(), y.value()));
          }
          if (y.has_value()) {
            return std::make_optional(y.value());
          }
          return std::optional<OVal>();
        });
  }
  partial_sums_scanned.pop_back();

  // this is step 7
  std::optional<OVal> sum_of_all_guys_before_my_rank;
  comm.scatter(std::span{partial_sums_scanned}, sum_of_all_guys_before_my_rank,
               0);

  // this is step 8
  if (!rng::empty(local_partial_sums) && !first_nonempty_segment_is_local) {
    OVal const sum_of_all_guys_before_my_rank_value =
        sum_of_all_guys_before_my_rank.value();
    std::for_each(std::execution::par_unseq,
                  rng::begin(local_partial_sums_scanned),
                  rng::end(local_partial_sums_scanned), [=](auto &&x) {
                    x = binary_op(sum_of_all_guys_before_my_rank_value, x);
                  });
  }

  // this is step 9
  auto local_partial_sum_iter = rng::begin(local_partial_sums_scanned);
  bool skip_nonempty_segment = first_nonempty_segment_is_local;
  bool nonempty_out_segment_already_found = false;
  for (auto &&out_seg : out_segments) {
    if (rng::empty(out_seg)) {
      continue;
    }

    auto first = ranges::local(rng::begin(out_seg));
    auto last = ranges::local(rng::end(out_seg));

    if (skip_nonempty_segment) {
      skip_nonempty_segment = false;
    } else {
      const OVal sum_of_all_guys_before_my_segment =
          nonempty_out_segment_already_found
              ? *local_partial_sum_iter
              : sum_of_all_guys_before_my_rank.value();
      events.push_back(for_each_local_async(
          dr::__detail::direct_iterator(first),
          dr::__detail::direct_iterator(last), [=](auto &&x) {
            x = binary_op(sum_of_all_guys_before_my_segment, x);
          }));
    }

    if (nonempty_out_segment_already_found) {
      ++local_partial_sum_iter;
    } else {
      nonempty_out_segment_already_found = true;
    }
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
