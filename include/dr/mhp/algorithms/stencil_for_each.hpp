// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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

/// Collective for_each on distributed range
template <typename... Ts>
void stencil_for_each(std::size_t radius, auto op,
                      dr::distributed_range auto &&dr1,
                      dr::distributed_range auto &&dr2) {
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();
  auto grid2 = dr2.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      auto t1 = grid1(tile_index, 0).mdspan();
      auto t2 = grid2(tile_index, 0).mdspan();

      // TODO support arbitrary ranks
      assert(t1.rank() == t2.rank() && t2.rank() == 2);

      // Do not update halo for first and last segment
      std::size_t first = 0 + radius * (tile_index == 0);
      std::size_t last =
          t1.extent(0) - radius * (tile_index == (grid1.extent(0) - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = radius; j < t1.extent(1) - radius; j++) {
          auto t1_stencil =
              md::mdspan(std::to_address(&t1(i, j)), t1.extents());
          auto t2_stencil =
              md::mdspan(std::to_address(&t2(i, j)), t2.extents());
          op(std::tuple(t1_stencil, t2_stencil));
        }
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
