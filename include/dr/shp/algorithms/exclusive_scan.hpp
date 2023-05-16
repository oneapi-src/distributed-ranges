// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <dr/shp/algorithms/inclusive_scan.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace dr::shp {

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename U, typename BinaryOp>
void exclusive_scan_impl_(ExecutionPolicy &&policy, R &&r, O &&o, U init,
                          BinaryOp &&binary_op) {
  using T = rng::range_value_t<O>;

  static_assert(
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  auto zipped_view = dr::shp::views::zip(r, o);
  auto zipped_segments = zipped_view.zipped_segments();

  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {

    U *d_inits = sycl::malloc_device<U>(rng::size(zipped_segments),
                                        shp::devices()[0], shp::context());

    std::vector<sycl::event> events;

    std::size_t segment_id = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto last_element = rng::prev(rng::end(__detail::local(in_segment)));
      auto dest = d_inits + segment_id;

      auto &&q = __detail::queue(dr::ranges::rank(in_segment));

      auto e = q.single_task([=] { *dest = *last_element; });
      events.push_back(e);
      segment_id++;
    }

    __detail::wait(events);
    events.clear();

    std::vector<U> inits(rng::size(zipped_segments));

    shp::copy(d_inits, d_inits + inits.size(), inits.data() + 1);

    fmt::print("Inits: {}\n", inits);

    sycl::free(d_inits, shp::context());

    inits[0] = init;

    auto root = dr::shp::devices()[0];
    dr::shp::device_allocator<T> allocator(dr::shp::context(), root);
    dr::shp::vector<T, dr::shp::device_allocator<T>> partial_sums(
        std::size_t(zipped_segments.size()), allocator);

    segment_id = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto &&q = __detail::queue(dr::ranges::rank(in_segment));
      auto &&local_policy = __detail::dpl_policy(dr::ranges::rank(in_segment));

      auto dist = rng::distance(in_segment);
      assert(dist > 0);

      auto first = rng::begin(in_segment);
      auto last = rng::end(in_segment);
      auto d_first = rng::begin(out_segment);

      auto init = inits[segment_id];

      auto event = oneapi::dpl::experimental::exclusive_scan_async(
          local_policy, dr::__detail::direct_iterator(first),
          dr::__detail::direct_iterator(last),
          dr::__detail::direct_iterator(d_first), init, binary_op);

      auto dst_iter = dr::ranges::local(partial_sums).data() + segment_id;

      auto src_iter = dr::ranges::local(out_segment).data();
      rng::advance(src_iter, dist - 1);

      auto e = q.submit([&](auto &&h) {
        h.depends_on(event);
        h.single_task([=]() {
          rng::range_value_t<O> value = *src_iter;
          *dst_iter = value;
        });
      });

      events.push_back(e);

      segment_id++;
    }

    __detail::wait(events);
    events.clear();

    auto &&local_policy = __detail::dpl_policy(0);

    auto first = dr::ranges::local(partial_sums).data();
    auto last = first + partial_sums.size();

    oneapi::dpl::experimental::inclusive_scan_async(local_policy, first, last,
                                                    first, binary_op)
        .wait();

    std::size_t idx = 0;
    for (auto &&segs : zipped_segments) {
      auto &&[in_segment, out_segment] = segs;

      auto &&local_policy = __detail::dpl_policy(dr::ranges::rank(out_segment));

      if (idx > 0) {
        T sum = partial_sums[idx - 1];

        auto first = rng::begin(out_segment);
        auto last = rng::end(out_segment);

        sycl::event e = oneapi::dpl::experimental::for_each_async(
            local_policy, dr::__detail::direct_iterator(first),
            dr::__detail::direct_iterator(last),
            [=](auto &&x) { x = binary_op(x, sum); });

        events.push_back(e);
      }
      idx++;
    }

    __detail::wait(events);

  } else {
    assert(false);
  }
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T, typename BinaryOp>
void exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init,
                    BinaryOp &&binary_op) {
  exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o), init,
                       std::forward<BinaryOp>(binary_op));
}

template <typename ExecutionPolicy, dr::distributed_contiguous_range R,
          dr::distributed_contiguous_range O, typename T>
void exclusive_scan(ExecutionPolicy &&policy, R &&r, O &&o, T init) {
  exclusive_scan_impl_(std::forward<ExecutionPolicy>(policy),
                       std::forward<R>(r), std::forward<O>(o), init,
                       std::plus<>{});
}

} // namespace dr::shp
