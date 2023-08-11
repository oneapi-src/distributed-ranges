// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/async>
#endif

#include <dr/detail/sycl_utils.hpp>

namespace dr::mhp::__detail {

namespace detail = dr::__detail;

}

namespace dr::mhp::__detail {

void local_inclusive_scan(auto policy, auto in, auto out, auto binary_op,
                          auto init, std::size_t seg_index) {
  auto in_begin_direct = detail::direct_iterator(in.begin());
  auto in_end_direct = detail::direct_iterator(in.end());
  auto out_begin_direct = detail::direct_iterator(out.begin());
  if (init && seg_index == 0) {
    std::inclusive_scan(policy, in_begin_direct, in_end_direct,
                        out_begin_direct, binary_op, init.value());
  } else {
    std::inclusive_scan(policy, in_begin_direct, in_end_direct,
                        out_begin_direct, binary_op);
  }
}

template <dr::distributed_contiguous_range R, dr::distributed_iterator O,
          typename BinaryOp, typename U = rng::range_value_t<R>>
auto inclusive_scan_impl_(R &&r, O &&d_first, BinaryOp &&binary_op,
                          std::optional<U> init = {}) {
  using value_type = U;
  assert(aligned(r, d_first));

  bool use_sycl = mhp::use_sycl();
  auto comm = default_comm();
  auto rank = comm.rank();
  auto local_segs = rng::views::zip(local_segments(r), local_segments(d_first));
  auto global_segs =
      rng::views::zip(dr::ranges::segments(r), dr::ranges::segments(d_first));
  std::size_t num_segs = std::size_t(rng::size(dr::ranges::segments(r)));

  // Pass 1 local inclusive scan
  std::size_t seg_index = 0;
  for (auto global_seg : global_segs) {
    auto [global_in, global_out] = global_seg;

    if (dr::ranges::rank(global_in) == rank) {
      auto local_in = dr::ranges::__detail::local(global_in);
      auto local_out = dr::ranges::__detail::local(global_out);
      if (use_sycl) {
#ifdef SYCL_LANGUAGE_VERSION
        local_inclusive_scan(dpl_policy(), local_in, local_out, binary_op, init,
                             seg_index);
#else
        assert(false);
#endif
      } else {
        local_inclusive_scan(std::execution::par_unseq, local_in, local_out,
                             binary_op, init, seg_index);
      }
    }

    seg_index++;
  }

  // Pass 2 put partial sums on root
  seg_index = 0;
  auto win = root_win();
  for (auto global_seg : global_segs) {
    // Do not need last segment
    if (seg_index == num_segs - 1) {
      break;
    }

    auto [global_in, global_out] = global_seg;
    if (dr::ranges::rank(global_in) == rank) {
      auto local_out = dr::ranges::__detail::local(global_out);
      auto back = use_sycl ? sycl_get(local_out.back()) : local_out.back();
      win.put(back, 0, seg_index);
    }

    seg_index++;
  }
  win.fence();

  // Pass 3: scan of partial sums on root
  if (rank == 0) {
    value_type *partials = win.local_data<value_type>();
    std::inclusive_scan(partials, partials + num_segs, partials, binary_op);
  }
  barrier();

  // Pass 4: rebase
  seg_index = 0;
  for (auto global_seg : global_segs) {
    if (seg_index > 0) {
      auto [global_in, global_out] = global_seg;

      auto offset = win.get<value_type>(0, seg_index - 1);
      auto rebase = [offset, binary_op](auto &v) { v = binary_op(v, offset); };
      if (dr::ranges::rank(global_in) == rank) {
        auto local_in = dr::ranges::__detail::local(global_in);
        auto local_out = rng::views::take(
            dr::ranges::__detail::local(global_out), rng::size(local_in));
        // dr::drlog.debug("rebase before: {}\n", local_out);
        if (use_sycl) {
#ifdef SYCL_LANGUAGE_VERSION
          auto wrap_rebase = [rebase, base = rng::begin(local_out)](auto idx) {
            rebase(base[idx]);
          };
          detail::parallel_for(dr::mhp::sycl_queue(), rng::distance(local_out),
                               wrap_rebase)
              .wait();
#else
          assert(false);
#endif
        } else {
          std::for_each(std::execution::par_unseq, local_out.begin(),
                        local_out.end(), rebase);
        }
        // dr::drlog.debug("rebase after: {}\n", local_out);
      }
    }

    seg_index++;
  }

  barrier();
  return d_first + rng::size(r);
}

} // namespace dr::mhp::__detail

namespace dr::mhp {

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
