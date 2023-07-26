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
#include <dr/detail/tuple_utils.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

namespace detail = dr::__detail;

/// Collective for_each on distributed range
template <typename... Ts>
void stencil_for_each(auto op, is_mdspan_view auto &&...drs) {
  auto ranges = std::tie(drs...);
  auto &&dr1 = std::get<0>(ranges);
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      // Calculate loop invariant info about the operands. Use a tuple
      // to hold the info for all operands.
      auto operand_infos =
          detail::tuple_transform(ranges, [tile_index](auto &&dr) {
            auto tile = dr.grid()(tile_index, 0);
            // mdspan for tile. This could be a submdspan, so we need the
            // extents of the root to get the memory strides
            return std::make_pair(tile.mdspan(), tile.root_mdspan().extents());
          });

      auto tile1 = grid1(tile_index, 0).mdspan();
      if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
        auto do_point = [=](auto index) {
          // Transform operand_infos into stencils
          auto stencils =
              detail::tuple_transform(operand_infos, [=](auto info) {
                return md::mdspan(
                    std::to_address(&info.first(index[0], index[1])),
                    info.second);
              });
          op(stencils);
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
          // Transform operand_infos into stencils
          auto stencils =
              detail::tuple_transform(operand_infos, [=](auto info) {
                return md::mdspan(std::to_address(&info.first(index)),
                                  info.second);
              });
          op(stencils);
        };
        detail::mdspan_foreach<tile1.rank(), decltype(invoke_index)>(
            tile1.extents(), invoke_index);
      }
    }
  }

  barrier();
}

/// Collective for_each on distributed range
template <typename... Ts> void for_each(auto op, is_mdspan_view auto &&...drs) {
  auto ranges = std::tie(drs...);
  auto &&dr1 = std::get<0>(ranges);
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      // make a tuple of mdspans
      auto operand_mdspans =
          detail::tuple_transform(ranges, [tile_index](auto &&dr) {
            auto tile = dr.grid()(tile_index, 0);
            return tile.mdspan();
          });

      auto tile1 = grid1(tile_index, 0).mdspan();
      if (mhp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
        //
        auto invoke_index = [=](auto index) {
          // Transform mdspans into references
          auto references = detail::tie_transform(
              operand_mdspans, [index](auto mdspan) -> decltype(auto) {
                return mdspan(index[0], index[1]);
              });
          op(references);
        };

        // TODO: Extend sycl_utils.hpp to handle ranges > 1D. It uses
        // ndrange and handles > 32 bits.
        dr::mhp::sycl_queue()
            .parallel_for(sycl::range(tile1.extent(0), tile1.extent(1)),
                          invoke_index)
            .wait();
#else
        assert(false);
#endif
      } else {
        // invoke op on a tuple of references created by using the mdspan's and
        // index
        auto invoke_index = [=](auto index) {
          // Transform operand_infos into references
          auto references = detail::tie_transform(
              operand_mdspans,
              [index](auto mdspan) -> decltype(auto) { return mdspan(index); });
          op(references);
        };
        detail::mdspan_foreach<tile1.rank(), decltype(invoke_index)>(
            tile1.extents(), invoke_index);
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
