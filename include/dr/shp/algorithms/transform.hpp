// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <dr/details/alignment.hpp>
#include <dr/shp/init.hpp>
#include <dr/shp/util.hpp>

namespace shp {

namespace __detail {
template <class T> T *get_local_pointer(shp::device_ptr<T> r) {
  return r.local();
}
template <class T> T *get_local_pointer(T *t) { return t; }
} // namespace __detail

void transform(auto &&policy, lib::distributed_range auto &&in,
               lib::distributed_iterator auto out, auto fn) {

  static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(policy)>, device_policy>);

  if (rng::empty(in))
    return;

  std::vector<sycl::event> events;
  auto &&devices = policy.get_devices();

  if (lib::__details::aligned(in.begin(), out)) {
    for (const auto &&[in_seg, out_seg] : rng::views::zip(
             lib::ranges::segments(in), lib::ranges::segments(out))) {

      auto local_in_seg = __detail::get_local_segment(in_seg);
      auto local_out_seg = __detail::get_local_segment(out_seg);

#ifdef DR_TESTING
      assert(drtesting::expect_fast_transform_path.value_or(true));
      assert(rng::size(in_seg) > 0);
      assert(rng::size(in_seg) <= rng::size(out_seg));
      assert(rng::size(local_in_seg) == rng::size(in_seg));
      assert(rng::size(local_out_seg) == rng::size(out_seg));
      assert(lib::ranges::rank(in_seg) == lib::ranges::rank(out_seg));
#endif

      events.emplace_back(
          sycl::queue(shp::context(), devices[lib::ranges::rank(in_seg)])
              .parallel_for(rng::size(local_in_seg), [=](auto idx) {
                local_out_seg[idx] = fn(local_in_seg[idx]);
              }));
    }
    __detail::wait(events);
  } else {

#ifdef DR_TESTING
    assert(!drtesting::expect_fast_transform_path.value_or(false));
#endif

    auto &&in_segments = lib::ranges::segments(in);
    auto in_segments_iter = rng::begin(in_segments);
    auto in_iter = std::begin(*in_segments_iter);

    std::vector<std::remove_cv_t<typename decltype(out)::value_type> *> buffers;

    for (auto &&out_seg_orig : lib::ranges::segments(out)) {
      auto out_device = devices[lib::ranges::rank(out_seg_orig)];
      auto out_seg = __detail::get_local_segment(out_seg_orig);
      auto out_iter = rng::begin(out_seg);
      while (out_iter != rng::end(out_seg)) {
        if (in_segments_iter == rng::end(in_segments))
          break;
        else if (in_iter == std::end(*in_segments_iter)) {
          ++in_segments_iter;
          in_iter = std::begin(*in_segments_iter);
        } else {
          const auto num_items =
              std::min(std::distance(in_iter, std::end(*in_segments_iter)),
                       std::distance(out_iter, rng::end(out_seg)));
          assert(num_items > 0);
          auto *buffer = sycl::malloc_device<std::remove_cvref_t<
              decltype(*__detail::get_local_pointer(in_iter))>>(
              num_items, out_device, shp::context());
          buffers.push_back(buffer);

          auto memcpy_event = sycl::queue(shp::context(), out_device)
                                  .copy(__detail::get_local_pointer(in_iter),
                                        buffer, num_items);

          events.emplace_back(
              sycl::queue(shp::context(), out_device).submit([&](auto &&h) {
                h.depends_on(memcpy_event);
                h.parallel_for(num_items, [=](auto idx) {
                  *(out_iter + idx) = fn(buffer[idx]);
                });
              }));

          std::advance(in_iter, num_items);
          std::advance(out_iter, num_items);
        }
      }
    }
    __detail::wait(events);
    for (auto *b : buffers)
      sycl::free(b, shp::context());
  }
}

} // namespace shp
