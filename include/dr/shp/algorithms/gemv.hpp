// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/containers/sparse_matrix.hpp>
#include <dr/shp/device_vector.hpp>
#include <dr/shp/distributed_span.hpp>

namespace shp {

template <lib::distributed_range C, typename T, typename I,
          lib::distributed_range B>
void gemv(C &&c, shp::sparse_matrix<T, I> &a, B &&b) {
  assert(c.size() == b.size());
  assert(a.shape()[1] == b.size());
  assert(a.grid_shape()[0] == c.segments().size());
  assert(a.grid_shape()[1] == 1);

  auto &&devices = shp::devices();

  using b_scalar_type = std::ranges::range_value_t<B>;

  using local_vector_type =
      shp::device_vector<b_scalar_type, shp::device_allocator<b_scalar_type>>;

  std::vector<local_vector_type> local_b;
  std::vector<sycl::event> copy_events;
  std::vector<sycl::event> comp_events;

  for (std::size_t i = 0; i < devices.size(); i++) {
    shp::device_allocator<T> allocator(shp::context(), devices[i]);
    local_b.push_back(local_vector_type(b.size(), allocator, i));
  }

  for (auto &&l_b : local_b) {
    auto event =
        shp::copy_async(b.begin(), b.end(), lib::ranges::local(l_b.begin()));
    copy_events.push_back(event);
  }

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    auto a_tile = a.tile({i, 0});

    auto a_iter = a_tile.begin();
    auto b_iter = lib::ranges::local(local_b[i].begin());
    auto c_iter = lib::ranges::local(c.segments()[i].begin());

    auto device = devices[a_tile.rank()];
    sycl::queue q(device);

    auto event = q.submit([=](auto &&h) {
      h.depends_on(copy_events[i]);
      h.parallel_for(a_tile.size(), [=](auto idx) {
        auto &&[index, a_v] = *(a_iter + idx);
        auto &&[i, k] = index;
        auto &&b_v = *(b_iter + k);
        auto &&c_v = *(c_iter + i);
        c_v += a_v * b_v;
      });
    });
    comp_events.push_back(event);
  }

  for (auto &&event : comp_events) {
    event.wait();
  }
}

} // namespace shp
