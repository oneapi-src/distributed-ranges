// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <experimental/mdspan>
namespace md = std::experimental;

template <typename T, typename Extents, typename LayoutPolicy,
          typename Accessor>
struct fmt::formatter<md::mdspan<T, Extents, LayoutPolicy, Accessor>, char>
    : public formatter<string_view> {
  template <typename FmtContext>
  auto format(md::mdspan<T, Extents, LayoutPolicy, Accessor> mdspan,
              FmtContext &ctx) const {
    std::array<std::size_t, mdspan.rank()> index;
    rng::fill(index, 0);
    format_mdspan(ctx, mdspan, index, 0);
    return ctx.out();
  }

  void format_mdspan(auto &ctx, auto mdspan, auto &index,
                     std::size_t dim) const {
    for (std::size_t i = 0; i < mdspan.extent(dim); i++) {
      index[dim] = i;
      if (dim == mdspan.rank() - 1) {
        if (i == 0) {
          format_to(ctx.out(), "{}: ", index);
        }
        format_to(ctx.out(), "{:4} ", mdspan(index));
      } else {
        format_mdspan(ctx, mdspan, index, dim + 1);
      }
    }
    format_to(ctx.out(), "\n");
  }
};

namespace dr::__detail {

template <std::size_t Rank> using dr_extents = std::array<std::size_t, Rank>;
template <std::size_t Rank> using md_extents = md::dextents<std::size_t, Rank>;

template <typename M, std::size_t Rank, std::size_t... indexes>
auto make_submdspan_impl(M mdspan, const dr_extents<Rank> &offset,
                         const dr_extents<Rank> &extents,
                         std::index_sequence<indexes...>) {
  return md::submdspan(
      mdspan,
      std::tuple(offset[indexes], offset[indexes] + extents[indexes])...);
}

// Mdspan accepts slices, but that is hard to work with because it
// requires parameter packs. Work with offset/size vectors internally
// and use slices at the interface
template <std::size_t Rank>
auto make_submdspan(auto mdspan, const std::array<std::size_t, Rank> &offset,
                    const std::array<std::size_t, Rank> &extents) {
  return make_submdspan_impl(mdspan, offset, extents,
                             std::make_index_sequence<Rank>{});
}

} // namespace dr::__detail
