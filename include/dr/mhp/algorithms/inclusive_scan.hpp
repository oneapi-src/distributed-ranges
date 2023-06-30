// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/async>
#endif

namespace dr::mhp {
namespace __detail {

template <typename R> std::size_t index_of_first_nonempty_segment(R &&dr) {
  for (auto &&[idx, s] :
       dr::ranges::segments(std::forward<R>(dr)) | rng::views::enumerate) {
    if (!rng::empty(s)) {
      return static_cast<std::size_t>(idx);
    }
  }
  return static_cast<std::size_t>(0);
}

template <typename R> std::size_t index_of_last_nonempty_segment(R &&dr) {
  for (auto &&[idx, s] : dr::ranges::segments(std::forward<R>(dr)) |
                             rng::views::enumerate | rng::views::reverse) {
    if (!rng::empty(s)) {
      return static_cast<std::size_t>(idx);
    }
  }
  return 0;
}

template <typename R> std::size_t non_empty_elements_count(R &&r) {
  std::size_t c = 0;
  for (auto &&e : r) {
    if (!rng::empty(e))
      ++c;
  }
  return c;
}

using WaitCallback = std::function<void()>;
WaitCallback wait_callback_from_event(auto &&e) {
  return WaitCallback([e] {
    auto e2 = e;
    e2.wait();
  });
}
inline WaitCallback noop_wait_callback() {
  return WaitCallback([] {});
}

template <class FunT, class... Init>
inline void inclusive_scan_local_on_cpu(auto &&first, auto &&last,
                                        auto &&dst_first, FunT &&func,
                                        Init &&...init) {
  std::inclusive_scan(std::execution::par_unseq, first, last, dst_first, func,
                      std::forward<Init>(init)...);
}

inline auto direct_local_iter(auto &&it) {
  return dr::__detail::direct_iterator(ranges::local(it));
}

template <class FunT, class... Init>
inline WaitCallback inclusive_scan_local_async(auto &&in_segment,
                                               auto &&out_segment, FunT &&func,
                                               Init &&...init) {
  auto first = direct_local_iter(rng::begin(in_segment));
  auto last = direct_local_iter(rng::end(in_segment));
  auto dst_first = direct_local_iter(rng::begin(out_segment));

#ifdef SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {
    return wait_callback_from_event(
        oneapi::dpl::experimental::inclusive_scan_async(
            dpl_policy(), first, last, dst_first, func,
            std::forward<Init>(init)...));
  }
#endif

  inclusive_scan_local_on_cpu(first, last, dst_first, func,
                              std::forward<Init>(init)...);
  return noop_wait_callback();
}

inline WaitCallback reduce_local_segment(auto &&s, auto result_iter,
                                         auto &&func) {
  // reduce has no BinaryOp without init signature, so we take first as init
  auto first = direct_local_iter(rng::begin(s));
  auto last = direct_local_iter(rng::end(s));
  auto init = *first++;
#ifdef SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {

    dr::drlog.debug("calling reduce_async on range:");
    for (auto &&x : s)
      dr::drlog.debug(" {}", x);
    dr::drlog.debug("\n");

    auto event = oneapi::dpl::experimental::reduce_async(dpl_policy(), first,
                                                         last, init, func);
    return [event, result_iter] {
      auto e = event;
      *result_iter = e.get();
      dr::drlog.debug("reduce_async finished, result:{}\n", *result_iter);
    };
  }
#endif

  *result_iter =
      std::reduce(std::execution::par_unseq, first, last, init, func);
  return noop_wait_callback();
}

inline void wait_for_events_and_clear(std::vector<WaitCallback> &events) {
  std::for_each(std::execution::par_unseq, rng::begin(events), rng::end(events),
                [](auto &&cb) { cb(); });
  events.clear();
}

// inclusive scan algorithm:
// let's denote segments as: rank(dot)segment_number
// let's segments be [1.1, 1.2, 2.1, 2.1, 3.1, 3.2]
// let's sum(X) denote collection of partial sum of range X
// let's s(X) denote just one-element sum of range X
// 1. (offloaded) for the first local segment compute its partial sum
//    sum(1.1) using initial value if provided
//    and for all other except last segments compute its one-value sum
//    rank:0 [s(1.2)]
//    rank:1 [s(2.1), s(2.2)]
//    rank:2 [s(3.1)]
//    first non-empty partial sum uses init value
// 2. get max (that is the last) element of the first local segment and all
//    prepend it to one-value sums of segments
//    rank:0 [max(sum(1.1), s(1.2)]
//    rank:1 [s(2.1), s(2.2)]
//    rank:2 [s(3.1)]
// 3. compute partial sum of above vector, e.g. on rank
//    rank:0 sum([max(sum(1.1), s(1.2)])
//    rank:1 sum([s(2.1), s(2.2)])
//    rank:2 sum([s(3.1)])
// 4. send max (that is the last element) of above sum to root process
//    let's denote above max of process no N as max_N
// 5. root process collects all maxes in a collection [max_1, max_2, max_3]
// 6. computes partial sum of it sum([max_1, max_2, max_3])
//    Let's denote this partial sums as a "big sum".
// 7. big sum is scattered to all processes
// 8. Nth process gets big_sum_N element and adds it to each element of local
//    partial sums computed in step 3
// 9. (offloaded) each (except very first) local i-th segment is modified by
//    computing partial sum of it with initial value being i-th partial sum
//    computed in the previous step
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
  auto in_segments = local_segments_with_idx(r);
  auto out_segments = local_segments(rng::subrange(d_first, d_last));

  std::vector<WaitCallback> events;
  using OVal = rng::iter_value_t<O>;

  const std::size_t first_nonempty_segment_idx =
      index_of_first_nonempty_segment(r);
  const std::size_t last_nonempty_segment_idx =
      index_of_last_nonempty_segment(r);

  std::vector<OVal> local_partial_sums(non_empty_elements_count(out_segments));
  auto local_partial_sums_it = rng::begin(local_partial_sums);

  // this is step 1
  for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

    auto &&[in_segment_with_idx, out_segment] = segs;
    auto &&[in_idx, in_segment] = in_segment_with_idx;

    if (rng::empty(in_segment))
      continue;

    if (in_idx == first_nonempty_segment_idx) {
      dr::drlog.debug("running scan idx:{}\n", static_cast<int>(in_idx));
      // first segment can be scanned without doing reduce first
      WaitCallback scan_wait_cb =
          init.has_value()
              ? inclusive_scan_local_async(in_segment, out_segment, binary_op,
                                           init.value())
              : inclusive_scan_local_async(in_segment, out_segment, binary_op);

      // save locally its sum from last element of output
      auto iter_to_last_out_element =
          direct_local_iter(rng::begin(out_segment));
      rng::advance(iter_to_last_out_element, rng::distance(in_segment) - 1);

      // this is step 2
      events.push_back(
          [scan_wait_cb, iter_to_last_out_element, local_partial_sums_it] {
            scan_wait_cb();
            *local_partial_sums_it = *iter_to_last_out_element;
          });
    } else if (in_idx != last_nonempty_segment_idx) {
      dr::drlog.debug("running reduce on idx:{}\n", static_cast<int>(in_idx));
      events.push_back(
          reduce_local_segment(in_segment, local_partial_sums_it, binary_op));
      // last segment is not needed to be reduced, all except last&first need to
    } else {
      dr::drlog.debug("running nothing idx:{}\n", static_cast<int>(in_idx));
    }

    ++local_partial_sums_it;
  }
  wait_for_events_and_clear(events);

  // this is step 3
  std::vector<OVal> local_partial_sums_scanned(rng::size(local_partial_sums));
  inclusive_scan_local_on_cpu(rng::begin(local_partial_sums),
                              local_partial_sums_it,
                              local_partial_sums_scanned.begin(), binary_op);

  dr::communicator &comm = default_comm();
  std::optional<OVal> local_partial_sum =
      rng::empty(local_partial_sums_scanned)
          ? std::optional<OVal>()
          : local_partial_sums_scanned.back();

  if (local_partial_sum.has_value())
    dr::drlog.debug("local partial sum computed is {}\n",
                    local_partial_sum.value());
  else
    dr::drlog.debug("no local partial sum\n");

  // below vector is used on 0 rank only but who cares
  std::vector<std::optional<OVal>> partial_sums(comm.size());

  // this is step 4 and 5
  comm.gather(local_partial_sum, std::span{partial_sums}, 0);

  // this is step 6
  std::vector<std::optional<OVal>> partial_sums_scanned(comm.size() + 1);
  if (comm.rank() == 0) {

    dr::drlog.debug("global rank=0 partial sums:");
    for (auto &&x : partial_sums)
      if (x.has_value())
        dr::drlog.debug(" {}", x.value());
      else
        dr::drlog.debug(" EMPTY");
    dr::drlog.debug("\n");

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

    dr::drlog.debug("global rank=0 partial sums scanned:");
    for (auto &&x : partial_sums_scanned)
      if (x.has_value())
        dr::drlog.debug(" {}", x.value());
      else
        dr::drlog.debug(" EMPTY");
    dr::drlog.debug("\n");
  }
  partial_sums_scanned.pop_back();

  // this is step 7
  std::optional<OVal> sum_of_all_guys_before_my_rank;
  comm.scatter(std::span{partial_sums_scanned}, sum_of_all_guys_before_my_rank,
               0);

  if (sum_of_all_guys_before_my_rank.has_value())
    dr::drlog.debug("sum_of_all_guys_before_my_rank:{}\n",
                    sum_of_all_guys_before_my_rank.value());
  else
    dr::drlog.debug("sum_of_all_guys_before_my_rank:NONE\n");

  // this is step 8
  if (!rng::empty(local_partial_sums) &&
      sum_of_all_guys_before_my_rank.has_value()) {
    std::for_each(std::execution::par_unseq,
                  rng::begin(local_partial_sums_scanned),
                  rng::end(local_partial_sums_scanned), [=](auto &&x) {
                    x = binary_op(sum_of_all_guys_before_my_rank.value(), x);
                  });
    dr::drlog.debug("local partial sums scan:");
    for (auto &&x : local_partial_sums_scanned)
      dr::drlog.debug(" {}", x);
    dr::drlog.debug("\n");
  } else
    dr::drlog.debug("local partial sums scan skipped\n");

  // this is step 9
  auto local_partial_sums_scanned_it = rng::begin(local_partial_sums_scanned);
  for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

    auto &&[in_segment_with_idx, out_segment] = segs;
    auto &&[in_idx, in_segment] = in_segment_with_idx;

    if (rng::empty(in_segment) || (in_idx == first_nonempty_segment_idx))
      continue;

    OVal init_of_segment;
    if (sum_of_all_guys_before_my_rank.has_value()) {
      init_of_segment = sum_of_all_guys_before_my_rank.value();
      sum_of_all_guys_before_my_rank = std::optional<OVal>();
    } else {
      init_of_segment = *local_partial_sums_scanned_it++;
    }
    events.push_back(inclusive_scan_local_async(in_segment, out_segment,
                                                binary_op, init_of_segment));
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
