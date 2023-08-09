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

  using value_type = U;
  auto comm = default_comm();
  bool first_rank = comm.rank() == 0;
  auto io_segments =
      rng::views::zip(local_segments(r), local_segments(d_first));
  std::vector<value_type> segment_bases;

  bool first_segment = true;
  for (auto io_segment : io_segments) {
    auto [in_segment, out_segment] = io_segment;
    dr::drlog.debug("in segment: {}\n", in_segment);
    if (init && first_rank && first_segment) {
      std::inclusive_scan(std::execution::par_unseq, in_segment.begin(),
                          in_segment.end(), out_segment.begin(), binary_op,
                          init.value());
    } else {
      std::inclusive_scan(std::execution::par_unseq, in_segment.begin(),
                          in_segment.end(), out_segment.begin(), binary_op);
    }
    segment_bases.push_back(out_segment.back());
    first_segment = false;
    dr::drlog.debug("out segment: {}\n", out_segment);
  }
  segment_bases.push_back(value_type{});

  dr::drlog.debug("segment bases in: {}\n", segment_bases);
  //  scan segment bases
  std::exclusive_scan(segment_bases.begin() + 1, segment_bases.end(),
                      segment_bases.begin() + 1, *segment_bases.begin(),
                      binary_op);
  dr::drlog.debug("segment bases out: {}\n", segment_bases);

  // gather ranks to root
  std::vector<value_type> rank_bases(comm.size() + 1);
  comm.gather(segment_bases.back(), std::span{rank_bases}, 0);

  dr::drlog.debug("rank bases in: {}\n", rank_bases);
  //  scan rank bases
  std::exclusive_scan(rank_bases.begin() + 1, rank_bases.end(),
                      rank_bases.begin() + 1, *rank_bases.begin(), binary_op);
  dr::drlog.debug("rank bases out: {}\n", rank_bases);

  // scatter rank bases
  value_type rank_base;
  comm.scatter(std::span{rank_bases}, rank_base, 0);

  // rebase local segments
  std::size_t i = 0;
  first_segment = true;
  for (auto io_segment : io_segments) {
    auto [in_segment, out_segment] = io_segment;
    auto segment = out_segment | rng::views::take(rng::size(in_segment));

    dr::drlog.debug("segment before rebase: {}\n", segment);
    auto base = rank_base;
    if (!first_rank || !first_segment) {
      if (!first_segment) {
        base = binary_op(base, segment_bases[i++]);
      }

      std::for_each(std::execution::par_unseq, rng::begin(segment),
                    rng::end(segment),
                    [binary_op, base](auto &v) { v = binary_op(v, base); });
    }
    first_segment = false;
    dr::drlog.debug("segment after rebase: {}\n", segment);
  }

  barrier();
  return d_first + rng::size(r);
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
