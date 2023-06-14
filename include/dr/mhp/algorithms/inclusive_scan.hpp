// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <optional>

// #include <sycl/sycl.hpp>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/async>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>

#include <algorithm>
#include <execution>
#include <type_traits>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

/// Collective iota on distributed range
// template <dr::distributed_range R, std::integral T> void iota(R &&r, T value) {
//   auto iota_view = rng::views::iota(value, T(value + rng::distance(r)));

//   for_each(views::zip(iota_view, r), [](auto &&elem) {
//     auto &&[idx, v] = elem;
//     v = idx;
//   });
// }

// /// Collective iota on iterator/sentinel for a distributed range
// template <dr::distributed_iterator Iter, std::integral T>
// void iota(Iter begin, Iter end, T value) {
//   auto r = rng::subrange(begin, end);
//   iota(r, value);
// }    

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp,
          typename U = rng::range_value_t<R>>
void inclusive_scan_impl_(ExecutionPolicy &&policy, R &&r, O &&o,
                          BinaryOp &&binary_op, std::optional<U> init = {}) {
    auto in_segments = local_segments(r);
    auto out_segments = local_segments(o);
    

    auto zipped_segments = views::zip(in_segments, out_segments);
    std::size_t segment_id = 0;
    for (auto &&segs : zipped_segments) {
        auto &&[in_segment, out_segment] = segs;
        auto dist = rng::distance(in_segment);
        assert(dist > 0);

        auto first = rng::begin(in_segment);
        auto last = rng::end(in_segment);
        auto d_first = rng::begin(out_segment);

        auto &&local_policy = __detail::dpl_policy(dr::ranges::rank(in_segment));


        if (segment_id == 0 && init.has_value()) {
           oneapi::dpl::experimental::inclusive_scan_async(
                local_policy, dr::__detail::direct_iterator(first),
                dr::__detail::direct_iterator(last),
                dr::__detail::direct_iterator(d_first), binary_op, init.value());
        } else {
            oneapi::dpl::experimental::inclusive_scan_async(
                local_policy, dr::__detail::direct_iterator(first),
                dr::__detail::direct_iterator(last),
                dr::__detail::direct_iterator(d_first), binary_op);
        }
        segment_id++;
    }
    barrier();
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp, typename T>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o,
                    BinaryOp &&binary_op, T init) {
  inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o),
                       std::forward<BinaryOp>(binary_op), std::optional(init));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o,
                    BinaryOp &&binary_op) {
  inclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o),
                       std::forward<BinaryOp>(binary_op));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O>
void inclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o) {
  inclusive_scan(std::forward<ExecutionPolicy>(policy), std::forward<R>(r),
                 std::forward<O>(o), std::plus<>());
}

// Distributed iterator versions

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter, typename BinaryOp, typename T>
OutputIter inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                          OutputIter d_first, BinaryOp &&binary_op, T init) {

  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  inclusive_scan(std::forward<ExecutionPolicy>(policy),
                 rng::subrange(first, last), rng::subrange(d_first, d_last),
                 std::forward<BinaryOp>(binary_op), init);

  return d_last;
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter, typename BinaryOp>
OutputIter inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                          OutputIter d_first, BinaryOp &&binary_op) {

  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  inclusive_scan(std::forward<ExecutionPolicy>(policy),
                 rng::subrange(first, last), rng::subrange(d_first, d_last),
                 std::forward<BinaryOp>(binary_op));

  return d_last;
}

template <typename ExecutionPolicy, dr::distributed_iterator Iter,
          dr::distributed_iterator OutputIter>
OutputIter inclusive_scan(ExecutionPolicy &&policy, Iter first, Iter last,
                          OutputIter d_first) {
  auto dist = rng::distance(first, last);
  auto d_last = d_first;
  rng::advance(d_last, dist);
  inclusive_scan(std::forward<ExecutionPolicy>(policy),
                 rng::subrange(first, last), rng::subrange(d_first, d_last));

  return d_last;
}

// Execution policy-less versions

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O>
void inclusive_scan(R &&r, O &&o) {
  inclusive_scan(std::execution::par_unseq, std::forward<R>(r), std::forward<O>(o));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp>
void inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op) {
  inclusive_scan(std::execution::par_unseq, std::forward<R>(r), std::forward<O>(o),
                 std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename BinaryOp, typename T>
void inclusive_scan(R &&r, O &&o, BinaryOp &&binary_op, T init) {
  inclusive_scan(std::execution::par_unseq, std::forward<R>(r), std::forward<O>(o),
                 std::forward<BinaryOp>(binary_op), init);
}

// Distributed iterator versions

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first) {
  return inclusive_scan(std::execution::par_unseq, first, last, d_first);
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename BinaryOp>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
                          BinaryOp &&binary_op) {
  return inclusive_scan(std::execution::par_unseq, first, last, d_first,
                        std::forward<BinaryOp>(binary_op));
}

template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
          typename BinaryOp, typename T>
OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
                          BinaryOp &&binary_op, T init) {
  return inclusive_scan(std::execution::par_unseq, first, last, d_first,
                        std::forward<BinaryOp>(binary_op), init);
}




// template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter>
// OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first) {
//   return inclusive_scan(dr::shp::par_unseq, first, last, d_first);
// }

// template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
//           typename BinaryOp>
// OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
//                           BinaryOp &&binary_op) {
//   return inclusive_scan(dr::shp::par_unseq, first, last, d_first,
//                         std::forward<BinaryOp>(binary_op));
// }

// template <dr::distributed_iterator Iter, dr::distributed_iterator OutputIter,
//           typename BinaryOp, typename T>
// OutputIter inclusive_scan(Iter first, Iter last, OutputIter d_first,
//                           BinaryOp &&binary_op, T init) {
//   return inclusive_scan(dr::shp::par_unseq, first, last, d_first,
//                         std::forward<BinaryOp>(binary_op), init);
// }

} // namespace dr::mhp
