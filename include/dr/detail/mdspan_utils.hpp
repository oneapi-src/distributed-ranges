// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>

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

//
// Mdspan accessor using an iterator
//
template <std::random_access_iterator Iter> class mdspan_iter_accessor {
public:
  using data_handle_type = Iter;
  using reference = std::iter_reference_t<Iter>;
  using offset_policy = mdspan_iter_accessor;

  constexpr mdspan_iter_accessor() noexcept = default;
  constexpr auto access(Iter iter, std::size_t index) const {
    return iter[index];
  }

  constexpr auto offset(Iter iter, std::size_t index) const noexcept {
    return iter + index;
  }
};

template <typename M, std::size_t Rank, std::size_t... indexes>
auto make_submdspan_impl(M mdspan, const dr_extents<Rank> &starts,
                         const dr_extents<Rank> &ends,
                         std::index_sequence<indexes...>) {
  return md::submdspan(mdspan, std::tuple(starts[indexes], ends[indexes])...);
}

// Mdspan accepts slices, but that is hard to work with because it
// requires parameter packs. Work with starts/size vectors internally
// and use slices at the interface
template <std::size_t Rank>
auto make_submdspan(auto mdspan, const std::array<std::size_t, Rank> &starts,
                    const std::array<std::size_t, Rank> &ends) {
  return make_submdspan_impl(mdspan, starts, ends,
                             std::make_index_sequence<Rank>{});
}

template <typename M>
void mdspan_pack_impl(M mdspan, auto &iter, dr_extents<M::rank()> &index,
                      std::size_t rank) {
  for (index[rank] = 0; index[rank] < mdspan.extent(rank); index[rank]++) {
    if (rank == M::rank() - 1) {
      *iter++ = mdspan(index);
    } else {
      mdspan_pack_impl(mdspan, iter, index, rank + 1);
    }
  }
}

template <typename M>
void mdspan_pack(M mdspan, std::forward_iterator auto iter) {
  dr_extents<M::rank()> index;
  mdspan_pack_impl(mdspan, iter, index, 0);
}

} // namespace dr::__detail
