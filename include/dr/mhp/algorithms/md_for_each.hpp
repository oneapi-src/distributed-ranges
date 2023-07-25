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
void stencil_for_each(auto op, is_mdspan_view auto &&...drs) {
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
        // Given an index, invoke op on a tuple of stencils
        auto invoke_index = [=](auto index) {
          // We have a tuple of infos and a index, invoke op on a
          // tuple of stencils
          auto invoke_op = [=](auto... infos) {
            op(std::tuple(md::mdspan(std::to_address(&infos.first(index)),
                                     infos.second)...));
          };
          std::apply(invoke_op, operand_infos);
        };
        dr::__detail::mdspan_foreach<tile1.rank(), decltype(invoke_index)>(
            tile1.extents(), invoke_index);
      }
    }
  }

  barrier();
}

/// Collective for_each on distributed range
template <typename... Ts> void for_each(auto op, is_mdspan_view auto &&...drs) {
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
      auto make_operand_mdspans = [=](auto &&dr) {
        auto tile = dr.grid()(tile_index, 0);
        return tile.mdspan();
      };
      // Calculate loop invariant info about the operands. Use a tuple
      // to hold the info for all operands.
      std::tuple operand_mdspans(make_operand_mdspans(drs)...);

      auto tile1 = grid1(tile_index, 0).mdspan();
      if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
        auto do_point = [=](auto index) {
          auto i = index[0];
          auto j = index[1];
          auto make_operands = [=](auto... mdspans) {
            return std::tie(mdspans(i, j)...);
          };
          op(std::apply(make_operands, operand_mdspans));
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
        // Given an index, invoke op on a tuple of references to mdspan element
        auto invoke_index = [=](auto index) {
          // We have a tuple of mdspans and a index, invoke op on a
          // tuple of references to mdspan elements
          auto invoke_op = [=](auto... mdspans) {
            op(std::tie(mdspans(index)...));
          };
          std::apply(invoke_op, operand_mdspans);
        };
        dr::__detail::mdspan_foreach<tile1.rank(), decltype(invoke_index)>(
            tile1.extents(), invoke_index);
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
