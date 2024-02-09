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

void local_exclusive_scan(auto policy, auto in, auto out, auto binary_op,
                          auto init, std::size_t seg_index) {
  auto in_begin_direct = detail::direct_iterator(in.begin());
  auto in_end_direct = detail::direct_iterator(in.end());
  auto out_begin_direct = detail::direct_iterator(out.begin());

  if (seg_index != 0) {
    assert(rng::size(in) > 1);
    assert(rng::size(out) > 1);
    --in_end_direct;
    ++out_begin_direct;
    std::inclusive_scan(policy, in_begin_direct, in_end_direct,
                        out_begin_direct, binary_op);
  } else {
    assert(init.has_value());
    std::exclusive_scan(policy, in_begin_direct, in_end_direct,
                        out_begin_direct, init.value(), binary_op);
  }
}

template <bool is_exclusive, dr::distributed_contiguous_range R,
          dr::distributed_iterator O, typename BinaryOp,
          typename U = rng::range_value_t<R>>
auto inclusive_exclusive_scan_impl_(R &&r, O &&d_first, BinaryOp &&binary_op,
                                    std::optional<U> init = {}) {
  using value_type = U;
  assert(aligned(r, d_first));

  bool use_sycl = mhp::use_sycl();
  auto comm = default_comm();

  // for input vector, which may have segment of size 1, do sequential scan
  if (rng::size(r) <= comm.size() * (comm.size() - 1) + 1) {
    std::vector<value_type> vec_in(rng::size(r));
    std::vector<value_type> vec_out(rng::size(r));
    mhp::copy(0, r, vec_in.begin());

    if (comm.rank() == 0) {
      if constexpr (is_exclusive) {
        assert(init.has_value());
        std::exclusive_scan(detail::direct_iterator(vec_in.begin()),
                            detail::direct_iterator(vec_in.end()),
                            detail::direct_iterator(vec_out.begin()),
                            init.value(), binary_op);
      } else {
        if (init.has_value()) {
          std::inclusive_scan(detail::direct_iterator(vec_in.begin()),
                              detail::direct_iterator(vec_in.end()),
                              detail::direct_iterator(vec_out.begin()),
                              binary_op, init.value());
        } else {
          std::inclusive_scan(detail::direct_iterator(vec_in.begin()),
                              detail::direct_iterator(vec_in.end()),
                              detail::direct_iterator(vec_out.begin()),
                              binary_op);
        }
      }
    }
    mhp::copy(0, vec_out, d_first);
    return d_first + rng::size(r);
  }

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
        if constexpr (is_exclusive) {
          local_exclusive_scan(dpl_policy(), local_in, local_out, binary_op,
                               init, seg_index);
        } else {
          local_inclusive_scan(dpl_policy(), local_in, local_out, binary_op,
                               init, seg_index);
        }
#else
        assert(false);
#endif
      } else {
        if constexpr (is_exclusive) {
          local_exclusive_scan(std::execution::par_unseq, local_in, local_out,
                               binary_op, init, seg_index);
        } else {
          local_inclusive_scan(std::execution::par_unseq, local_in, local_out,
                               binary_op, init, seg_index);
        }
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
      auto local_in = dr::ranges::__detail::local(global_in);
      rng::range_value_t<R> back;
      if constexpr (is_exclusive) {
        if (use_sycl) {
          auto ret = sycl_get(local_out.back(), local_in.back());
          back = binary_op(ret.first, ret.second);
        } else {
          back = binary_op(local_out.back(), local_in.back());
        }
      } else {
        back = use_sycl ? sycl_get(local_out.back()) : local_out.back();
      }

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
        auto local_out_adj = [use_sycl](auto local_out, auto offset) {
          bool _use_sycl = use_sycl;
          if constexpr (is_exclusive) {
            auto local_out_begin_direct =
                detail::direct_iterator(local_out.begin());
            if (_use_sycl) {
              sycl_copy(&offset, &(*local_out_begin_direct));
            } else {
              *local_out_begin_direct = offset;
            }
            return local_out | rng::views::drop(1);
          } else {
            return local_out;
          }
        }(local_out, offset);
        if (use_sycl) {
#ifdef SYCL_LANGUAGE_VERSION
          auto wrap_rebase = [rebase, base = rng::begin(local_out_adj)](
                                 auto idx) { rebase(base[idx]); };
          detail::parallel_for(dr::mhp::sycl_queue(),
                               sycl::range<>(rng::distance(local_out_adj)),
                               wrap_rebase)
              .wait();
#else
          assert(false);
#endif
        } else {
          std::for_each(std::execution::par_unseq, local_out_adj.begin(),
                        local_out_adj.end(), rebase);
        }
        // dr::drlog.debug("rebase after: {}\n", local_out_adj);
      }
    }
    seg_index++;
  }

  barrier();
  return d_first + rng::size(r);
}
} // namespace dr::mhp::__detail
