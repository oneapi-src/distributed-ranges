// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/algorithms/matrix/local_gemm.hpp>
#include <dr/shp/containers/distributed_dense_matrix.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

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

template <typename T>
void gemm_spmd(distributed_dense_matrix<T> &a, distributed_dense_matrix<T> &b,
               distributed_dense_matrix<T> &c) {
  // Matrix dimensions must match (algorithm requirement)
  assert(c.shape()[0] == a.shape()[0]);
  assert(c.shape()[1] == b.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  // Tile grid dimensions must match (implementation limitation)

  assert(c.grid_shape()[0] == a.grid_shape()[0]);
  assert(c.grid_shape()[1] == b.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<std::thread> threads;

  for (std::size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (std::size_t j = 0; j < c.grid_shape()[1]; j++) {
      auto c_local = c.tile({i, j});

      auto &&q = __detail::queue(dr::ranges::rank(c_local));

      threads.emplace_back([c_local, i, j, &q, &a, &b, &c] {
        dr::shp::device_allocator<T> allocator(q);
        std::size_t k_offset = i + j;

        auto a_f = a.get_tile_async({i, k_offset % a.grid_shape()[1]});
        auto b_f = b.get_tile_async({k_offset % a.grid_shape()[1], j});

        for (std::size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          std::size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto a_tile = a_f.get();
          auto b_tile = b_f.get();

          fmt::print("Multiplying a[{}, {}] x b[{}, {}]. Dimensions {} x {}, "
                     "{} x {}\n",
                     i, k, k, j, a_tile.shape()[0], a_tile.shape()[1],
                     b_tile.shape()[0], b_tile.shape()[1]);

          dr::shp::dense_matrix_view a_local(a_tile);
          dr::shp::dense_matrix_view b_local(b_tile);

          if (k_ + 1 < a.grid_shape()[1]) {
            a_f = a.get_tile_async({i, (k + 1) % a.grid_shape()[1]});
            b_f = b.get_tile_async({(k + 1) % a.grid_shape()[1], j});
          }

          fmt::print("Calling local_gemm...\n");
          __detail::local_gemm(q, __detail::local(a_local),
                               __detail::local(b_local),
                               __detail::local(c_local))
              .wait();
          fmt::print("After local_gemm...\n");
        }
      });
    }
  }

  fmt::print("Joining...\n");

  for (auto &&t : threads) {
    t.join();
  }
}

} // namespace dr::shp
