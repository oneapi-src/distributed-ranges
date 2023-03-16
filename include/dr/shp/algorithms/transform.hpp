// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <dr/shp/init.hpp>
#include <dr/shp/util.hpp>

namespace shp {

namespace __detail {
template <class T> T *get_local_pointer(shp::device_ptr<T> r) {
  return r.local();
}
template <class T> T *get_local_pointer(T *t) { return t; }
} // namespace __detail

template <class ExecutionPolicy>
void transform(ExecutionPolicy &&policy, lib::distributed_range auto &&in,
               lib::distributed_iterator auto out, auto && fn) {

  static_assert( // currently only one policy supported
      std::is_same_v<std::remove_cvref_t<ExecutionPolicy>, device_policy>);

  std::vector<sycl::event> events;
  using OutT = typename decltype(out)::value_type;
  std::vector<device_vector<OutT, device_allocator<OutT>>*> buffers;

  for (auto&& [in_seg, out_seg] : views::zip(in, rng::subrange(out, out+rng::size(in))).zipped_segments()) {

    sycl::queue q(shp::context(), policy.get_devices()[in_seg.rank()]);
    const std::size_t seg_size = rng::size(in_seg);
    assert(seg_size == rng::size(out_seg));
    auto local_in_seg = __detail::get_local_segment(in_seg);

    if (in_seg.rank() == out_seg.rank()) {
      auto local_out_seg = __detail::get_local_segment(out_seg);
      events.emplace_back(
          q.parallel_for(seg_size, [=](auto idx) { local_out_seg[idx] = fn(local_in_seg[idx]); }));
    } else {

      auto* buffer = new device_vector(seg_size, device_allocator<OutT>(q), in_seg.rank());
      buffers.push_back(buffer);

      sycl::event compute_event =
          q.parallel_for(seg_size, [=](auto idx) { (*buffer)[idx] = fn(local_in_seg[idx]); });
      events.emplace_back(q.copy(__detail::get_local_pointer(buffer->begin()), __detail::get_local_pointer(out_seg.begin()), seg_size, compute_event));
    }
  }
  __detail::wait(events);
  
  for (auto* b : buffers) delete b;
}

} // namespace shp
