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
  auto &&dr0 = std::get<0>(ranges);
  if (rng::empty(dr0)) {
    return;
  }

  auto all_segments = rng::views::zip(dr::ranges::segments(drs)...);
  for (auto segs : all_segments) {
    auto seg0 = std::get<0>(segs);
    auto mdspan0 = seg0.mdspan();

    // If local
    if (dr::ranges::rank(seg0) == default_comm().rank()) {
      // Calculate loop invariant info about the operands. Use a tuple
      // to hold the info for all operands.
      auto operand_infos = detail::tuple_transform(segs, [](auto &&seg) {
        // mdspan for tile. This could be a submdspan, so we need the
        // extents of the root to get the memory strides
        return std::make_pair(seg.mdspan(), seg.root_mdspan().extents());
      });

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
            .parallel_for(sycl::range(mdspan0.extent(0), mdspan0.extent(1)),
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
#if 0
        // Does not vectorize. Something about loop index being forced into memory
        detail::mdspan_foreach<mdspan0.rank(), decltype(invoke_index)>(
            mdspan0.extents(), invoke_index);
#else
        for (std::size_t i = 0; i < mdspan0.extents().extent(0); i++) {
          for (std::size_t j = 0; j < mdspan0.extents().extent(1); j++) {
            invoke_index(std::array<std::size_t, 2>{i, j});
          }
        }
#endif
      }
    }
  }

  barrier();
}

/// Collective for_each on distributed range
template <typename... Ts> void for_each(auto op, is_mdspan_view auto &&...drs) {
  auto ranges = std::tie(drs...);
  auto &&dr0 = std::get<0>(ranges);
  if (rng::empty(dr0)) {
    return;
  }

  auto all_segments = rng::views::zip(dr::ranges::segments(drs)...);
  for (auto segs : all_segments) {
    auto seg0 = std::get<0>(segs);
    auto mdspan0 = seg0.mdspan();

    // If local
    if (dr::ranges::rank(seg0) == default_comm().rank()) {
      // make a tuple of mdspans
      auto operand_mdspans = detail::tuple_transform(
          segs, [](auto &&seg) { return seg.mdspan(); });

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
            .parallel_for(sycl::range(mdspan0.extent(0), mdspan0.extent(1)),
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
        detail::mdspan_foreach<mdspan0.rank(), decltype(invoke_index)>(
            mdspan0.extents(), invoke_index);
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
