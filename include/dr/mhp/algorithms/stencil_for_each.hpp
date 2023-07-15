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

/// Collective for_each on distributed range
template <typename... Ts>
void stencil_for_each(auto op, dr::distributed_range auto &&...drs) {
  auto &&dr1 = std::get<0>(std::tie(drs...));
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      auto make_operand_info = [=](auto &&dr) {
        auto tile = dr.grid()(tile_index, 0);
        // mdspan for tile. This could be a submdspan, so we need the
        // extents of the root to get the memory strides
        return std::pair(tile.mdspan(), tile.root_mdspan().extents());
      };
      // Calculate loop invariant info about the operands. Use a tuple
      // to hold the info for all operands.
      std::tuple operand_infos(make_operand_info(drs)...);

      auto tile1 = grid1(tile_index, 0).mdspan();
      if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
        auto do_point = [=](auto index) {
          auto i = index[0];
          auto j = index[1];
          auto make_operands = [=](auto... infos) {
            // Use mdspan for tile to calculate the address of the
            // current element, and make an mdspan centered on it with
            // strides from the root mdspan
            return std::tuple(md::mdspan(std::to_address(&infos.first(i, j)),
                                         infos.second)...);
          };
          op(std::apply(make_operands, operand_infos));
        };
        // TODO: Extend sycl_utils.hpp to handle ranges > 1D. It uses
        // ndrange and handles > 32 bits.
        dr::mhp::sycl_queue()
            .parallel_for(sycl::range(tile1.extent(0), tile1.extent(1)),
                          do_point)
            .wait();
#else
        assert(false);
#endif
      } else {
        for (std::size_t i = 0; i < tile1.extent(0); i++) {
          for (std::size_t j = 0; j < tile1.extent(1); j++) {
            auto make_operands = [=](auto... infos) {
              // Use mdspan for tile to calculate the address of the
              // current element, and make an mdspan centered on it with
              // strides from the root mdspan
              return std::tuple(md::mdspan(std::to_address(&infos.first(i, j)),
                                           infos.second)...);
            };
            op(std::apply(make_operands, operand_infos));
          }
        }
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
