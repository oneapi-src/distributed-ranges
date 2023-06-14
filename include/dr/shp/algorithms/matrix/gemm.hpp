// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/algorithms/matrix/local_gemm.hpp>
#include <dr/shp/containers/distributed_dense_matrix.hpp>

namespace dr::shp {

template <typename T>
void gemm(distributed_dense_matrix<T> &a, distributed_dense_matrix<T> &b,
          distributed_dense_matrix<T> &c) {
  // Matrix dimensions must match (algorithm requirement)
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  // Tile grid dimensions must match (implementation limitation)

  assert(c.grid_shape()[0] == a.grid_shape()[0]);
  assert(c.grid_shape()[1] == b.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<sycl::event> events;
  events.reserve(c.grid_shape()[0] * c.grid_shape()[1] * a.grid_shape()[1]);

  for (std::size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (std::size_t j = 0; j < c.grid_shape()[1]; j++) {
      // For each tile of the output C matrix
      auto &&c_tile = c.tile({i, j});

      std::vector<sycl::event> local_events;
      local_events.reserve(a.grid_shape()[1]);

      std::size_t k_offset = i + j;
      for (std::size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
        std::size_t k = (k_ + k_offset) % a.grid_shape()[1];

        auto &&a_tile = a.tile({i, k});
        auto &&b_tile = b.tile({k, j});

        auto &&q = __detail::queue(dr::ranges::rank(c_tile));

        auto e = __detail::local_gemm(q, __detail::local(a_tile),
                                      __detail::local(b_tile),
                                      __detail::local(c_tile), local_events);

        local_events.push_back(e);
      }

      for (auto &&e : local_events) {
        events.push_back(e);
      }
    }
  }

  __detail::wait(events);
}

} // namespace dr::shp
