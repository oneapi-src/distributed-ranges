// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>

namespace dr::__detail {

inline std::size_t round_up(std::size_t n, std::size_t multiple) {
  if (multiple == 0)
    return n;

  int remainder = n % multiple;
  if (remainder == 0)
    return n;

  return n + multiple - remainder;
}

template <typename Mdspan>
concept mdspan_like = requires(Mdspan &mdspan) {
  mdspan.rank();
  mdspan.extents();
};

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

template <std::size_t Rank, typename Op>
void mdspan_foreach(md_extents<Rank> extents, Op op,
                    dr_extents<Rank> index = dr_extents<Rank>(),
                    std::size_t rank = 0) {
  for (index[rank] = 0; index[rank] < extents.extent(rank); index[rank]++) {
    if (rank == Rank - 1) {
      op(index);
    } else {
      mdspan_foreach(extents, op, index, rank + 1);
    }
  }
}

// Pack mdspan into contiguous container
void mdspan_copy(mdspan_like auto src, std::forward_iterator auto dst) {
  auto pack = [src, &dst](auto index) { *dst++ = src(index); };
  mdspan_foreach<src.rank(), decltype(pack)>(src.extents(), pack);
}

// unpack contiguous container into mdspan
void mdspan_copy(std::forward_iterator auto src, mdspan_like auto dst) {
  auto unpack = [&src, dst](auto index) { dst(index) = *src++; };
  mdspan_foreach<dst.rank(), decltype(unpack)>(dst.extents(), unpack);
}

// copy mdspan to mdspan
void mdspan_copy(mdspan_like auto src, mdspan_like auto dst) {
  assert(src.extents() == dst.extents());
  auto copy = [src, dst](auto index) { dst(index) = src(index); };
  mdspan_foreach<src.rank(), decltype(copy)>(src.extents(), copy);
}

// For operator(), rearrange indices according to template arguments.
//
// For mdtranspose<mdspan2d, 1, 0> a(b);
//
// a(3, 4) will do b(4, 3)
//
template <typename Mdspan, std::size_t... Is>
class mdtranspose : public Mdspan {
public:
  // Inherit constructors from base class
  mdtranspose(Mdspan &mdspan) : Mdspan(mdspan) {}

  // rearrange indices according to template arguments
  template <std::integral... Indexes>
  auto &operator()(Indexes... indexes) const {
    std::tuple index(indexes...);
    return Mdspan::operator()(std::get<Is>(index)...);
  }
  auto &operator()(std::array<std::size_t, Mdspan::rank()> index) const {
    return Mdspan::operator()(index[Is]...);
  }

  auto extents() const {
    return md_extents<Mdspan::rank()>(Mdspan::extent(Is)...);
  }
  auto extent(std::size_t d) const { return extents().extent(d); }
};

} // namespace dr::__detail

template <dr::__detail::mdspan_like Mdspan>
struct fmt::formatter<Mdspan, char> : public formatter<string_view> {
  template <typename FmtContext>
  auto format(Mdspan mdspan, FmtContext &ctx) const {
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
