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

template <typename InitT>
inline WaitCallback reduce_local_segment(auto &&s, auto result_iter,
                                         auto &&func,
                                         std::optional<InitT> init_opt) {
  // reduce has no BinaryOp without init signature, so we take first as init if
  // there is no init provided in init_opt
  auto first = direct_local_iter(rng::begin(s));
  auto last = direct_local_iter(rng::end(s));
  auto init = init_opt.has_value() ? init_opt.value() : *first++;
  if (first == last) {
    *result_iter = init;
    return noop_wait_callback();
  }
#ifdef SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {
    auto event = oneapi::dpl::experimental::reduce_async(dpl_policy(), first,
                                                         last, init, func);
    return [event, result_iter] {
      auto e = event;
      *result_iter = e.get();
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
// 1. (offloaded) for all segments except last one compute its one-value sum
//    rank:0 [s(1.1), s(1.2)]
//    rank:1 [s(2.1), s(2.2)]
//    rank:2 [s(3.1)]
//    first non-empty partial sum uses init value
// 2. compute partial sum of above vector, e.g. on rank
//    rank:0 sum([s(1.1), s(1.2)])
//    rank:1 sum([s(2.1), s(2.2)])
//    rank:2 sum([s(3.1)])
// 3. send max (that is the last element) of above sum to root process
//    let's denote above max of process no N as max_N
// 4. root process collects all maxes in a collection [max_1, max_2, max_3]
// 5. computes partial sum of it sum([max_1, max_2, max_3])
//    Let's denote this partial sums as a "big sum".
// 6. big sum is scattered to all processes
// 7. Nth process gets big_sum_N element and adds it to each element of local
//    partial sums computed in step 3
// 8. (offloaded) each local i-th in-segment is rewritten to i-th out-segment by
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

  if (first_nonempty_segment_idx == last_nonempty_segment_idx) {
    // just do one local inclusive scan
    for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

      auto &&[in_segment_with_idx, out_segment] = segs;
      auto &&[in_idx, in_segment] = in_segment_with_idx;

      if (rng::empty(in_segment))
        continue;

      assert(in_idx == first_nonempty_segment_idx);
      if (init.has_value()) {
        inclusive_scan_local_async(in_segment, out_segment, binary_op,
                                   init.value())();
      } else {
        inclusive_scan_local_async(in_segment, out_segment, binary_op)();
      }
    }
    return d_last;
  }

  std::vector<OVal> local_partial_sums(non_empty_elements_count(out_segments));
  auto local_partial_sums_it = rng::begin(local_partial_sums);

  // this is step 1
  for (auto &&[in_idx, in_segment] : in_segments) {

    if (rng::empty(in_segment))
      continue;

    if (in_idx != last_nonempty_segment_idx) {
      events.push_back(reduce_local_segment(
          in_segment, local_partial_sums_it, binary_op,
          init.has_value() && first_nonempty_segment_idx == in_idx
              ? init
              : std::optional<OVal>()));
      // last segment is not needed to be reduced
    }

    ++local_partial_sums_it;
  }
  wait_for_events_and_clear(events);

  // this is step 2
  std::vector<OVal> local_partial_sums_scanned(rng::size(local_partial_sums));
  inclusive_scan_local_on_cpu(rng::begin(local_partial_sums),
                              local_partial_sums_it,
                              local_partial_sums_scanned.begin(), binary_op);

  dr::communicator &comm = default_comm();
  std::optional<OVal> local_partial_sum =
      rng::empty(local_partial_sums_scanned)
          ? std::optional<OVal>()
          : local_partial_sums_scanned.back();

  // below vector is used on 0 rank only but who cares
  std::vector<std::optional<OVal>> partial_sums(comm.size());

  // these are steps 3 and 4
  comm.gather(local_partial_sum, std::span{partial_sums}, 0);

  // this is step 5
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

  // this is step 6
  std::optional<OVal> sum_of_all_guys_before_my_rank;
  comm.scatter(std::span{partial_sums_scanned}, sum_of_all_guys_before_my_rank,
               0);

  // this is step 7
  if (!rng::empty(local_partial_sums) &&
      sum_of_all_guys_before_my_rank.has_value()) {
    std::for_each(std::execution::par_unseq,
                  rng::begin(local_partial_sums_scanned),
                  rng::end(local_partial_sums_scanned), [=](auto &&x) {
                    x = binary_op(sum_of_all_guys_before_my_rank.value(), x);
                  });
  }

  // this is step 8
  auto local_partial_sums_scanned_it = rng::begin(local_partial_sums_scanned);
  for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

    auto &&[in_segment_with_idx, out_segment] = segs;
    auto &&[in_idx, in_segment] = in_segment_with_idx;

    if (rng::empty(in_segment))
      continue;

    if (in_idx == first_nonempty_segment_idx && !init.has_value()) {
      events.push_back(
          inclusive_scan_local_async(in_segment, out_segment, binary_op));
    } else {
      OVal init_of_segment;
      if (in_idx == first_nonempty_segment_idx) {
        init_of_segment = init.value();
        assert(!sum_of_all_guys_before_my_rank.has_value());
      } else if (sum_of_all_guys_before_my_rank.has_value()) {
        init_of_segment = sum_of_all_guys_before_my_rank.value();
        sum_of_all_guys_before_my_rank = std::optional<OVal>();
      } else {
        init_of_segment = *local_partial_sums_scanned_it++;
      }

      events.push_back(inclusive_scan_local_async(in_segment, out_segment,
                                                  binary_op, init_of_segment));
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
