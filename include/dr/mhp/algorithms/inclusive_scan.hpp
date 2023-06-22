// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/async>
#endif

namespace dr::mhp {

namespace __detail {

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

#ifdef SYCL_LANGUAGE_VERSION
  std::vector<sycl::event> events;
#endif

  using OVal = rng::iter_value_t<O>;
  // how to get allocator used with input and output range? I assume it a
  // default one probably we should forbid using non-default ones with
  // distributed_ranges containers (like distributed_vector)
  OVal *local_partial_sums =
      local_segments_count
          ? default_allocator<OVal>().allocate(local_segments_count)
          : nullptr;
  OVal *local_partial_sums_iter = local_partial_sums;

  for (auto &&segs : rng::views::zip(in_segments, out_segments)) {

    auto &&[in_segment, out_segment] = segs;

    auto dist = rng::distance(in_segment);
    if (dist == 0) {
      continue;
    }

    auto first = ranges::local(rng::begin(in_segment));
    auto last = ranges::local(rng::end(in_segment));
    auto s_first = ranges::local(rng::begin(out_segment));

#ifdef SYCL_LANGUAGE_VERSION
    sycl::event one_segment_scan_event;
    if (mhp::use_sycl()) {
      one_segment_scan_event = oneapi::dpl::experimental::inclusive_scan_async(
          dpl_policy(), dr::__detail::direct_iterator(first),
          dr::__detail::direct_iterator(last),
          dr::__detail::direct_iterator(s_first), binary_op);
    } else
#endif
      std::inclusive_scan(std::execution::par_unseq, first, last, s_first,
                          binary_op);

    rng::advance(s_first, dist - 1);

#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl()) {
      events.push_back(mhp::sycl_queue().submit([&](auto &&h) {
        h.depends_on(one_segment_scan_event);
        h.single_task([=]() {
          OVal tmp = *s_first;
          *local_partial_sums_iter = tmp;
        });
      }));
    } else
#endif
      *local_partial_sums_iter = *s_first;
    ++local_partial_sums_iter;
  }

#ifdef SYCL_LANGUAGE_VERSION
  sycl::event::wait(events);
  events.clear(); // events are reused later in this function
#endif

  const std::size_t local_partial_sums_count =
      rng::distance(local_partial_sums, local_partial_sums_iter);

  OVal *local_partial_sums_scanned =
      local_partial_sums_count
          ? default_allocator<OVal>().allocate(local_partial_sums_count)
          : nullptr;

  if (local_partial_sums_count) {
#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl())
      oneapi::dpl::experimental::inclusive_scan_async(
          dpl_policy(), local_partial_sums,
          local_partial_sums + local_partial_sums_count,
          local_partial_sums_scanned, binary_op)
          .wait();
    else
#endif
      std::inclusive_scan(std::execution::par_unseq, local_partial_sums,
                          local_partial_sums + local_partial_sums_count,
                          local_partial_sums_scanned, binary_op);
  }

  dr::communicator &comm = default_comm();
  std::optional<OVal> local_partial_sum =
      local_partial_sums_count
          ? local_partial_sums_scanned[local_partial_sums_count - 1]
          : std::optional<OVal>();

  // below vector is used on 0 rank only but who cares
  std::vector<std::optional<OVal>> partial_sums(comm.size()); // dr-style ignore

  comm.gather(local_partial_sum, partial_sums, 0);

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

  // scatter partial_sums_scanned
  std::optional<OVal> sum_of_all_guys_before_my_rank;
  comm.scatter(partial_sums, sum_of_all_guys_before_my_rank, 0);

  if (local_partial_sums_count && sum_of_all_guys_before_my_rank.has_value()) {
    const OVal sum_of_all_guys_before_my_rank_value =
        sum_of_all_guys_before_my_rank.value();
#ifdef SYCL_LANGUAGE_VERSION
    if (mhp::use_sycl())
      oneapi::dpl::experimental::for_each_async(
          dpl_policy(), local_partial_sums_scanned,
          local_partial_sums_scanned + local_partial_sums_count,
          [=](auto &&x) {
            x = binary_op(sum_of_all_guys_before_my_rank_value, x);
          })
          .wait();
    else
#endif
      std::for_each(std::execution::par_unseq, local_partial_sums_scanned,
                    local_partial_sums_scanned + local_partial_sums_count,
                    [=](auto &&x) {
                      x = binary_op(sum_of_all_guys_before_my_rank_value, x);
                    });
  }

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
              ? local_partial_sums_scanned[local_partial_sum_idx - 1]
              : sum_of_all_guys_before_my_rank.value();
#ifdef SYCL_LANGUAGE_VERSION
      if (mhp::use_sycl()) {
        events.push_back(oneapi::dpl::experimental::for_each_async(
            dpl_policy(), dr::__detail::direct_iterator(first),
            dr::__detail::direct_iterator(last), [=](auto &&x) {
              x = binary_op(sum_of_all_guys_before_my_segment, x);
            }));
      } else
#endif
        std::for_each(std::execution::par_unseq, first, last, [=](auto &&x) {
          x = binary_op(sum_of_all_guys_before_my_segment, x);
        });
    }
    ++local_partial_sum_idx;
    nonempty_out_segment_already_found = true;
  }

#ifdef SYCL_LANGUAGE_VERSION
  sycl::event::wait(events);
#endif

  if (local_partial_sums_count)
    default_allocator<OVal>().deallocate(local_partial_sums,
                                         local_segments_count);
  default_allocator<OVal>().deallocate(local_partial_sums_scanned,
                                       local_partial_sums_count);
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
