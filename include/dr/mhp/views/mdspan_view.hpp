// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr::mhp::decomp {

inline constexpr std::size_t div = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t all = div - 1;

} // namespace dr::mhp::decomp

namespace dr::mhp {

//
// Add a local mdspan to the underlying segment
//
template <typename BaseSegment, std::size_t Rank,
          typename Layout = md::layout_right>
class mdsegment : public BaseSegment {
public:
  mdsegment(std::size_t index, BaseSegment segment,
            dr::__detail::dr_extents<Rank> tile_offsets,
            dr::__detail::dr_extents<Rank> tile_extents)
      : BaseSegment(segment), index_(index),
        mdspan_(local_tile(segment, tile_offsets, tile_extents)) {}

  auto mdspan() const { return mdspan_; }
  auto index() const { return index_; }

private:
  using T = rng::range_value_t<BaseSegment>;

  static auto local_tile(BaseSegment segment,
                         const dr::__detail::dr_extents<Rank> &tile_offsets,
                         const dr::__detail::dr_extents<Rank> &tile_extents) {
    // Undefined behavior if the segments is not local
    T *ptr = dr::ranges::rank(segment) == default_comm().rank()
                 ? std::to_address(dr::ranges::local(rng::begin(segment)))
                 : nullptr;
    auto mdspan = md::mdspan(ptr, tile_extents);
    return dr::__detail::make_submdspan(mdspan, tile_offsets, tile_extents);
  }

  std::size_t index_;
  md::mdspan<T, dr::__detail::md_extents<Rank>, md::layout_right> mdspan_;
};

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

//
// Wrap a distributed range, adding an mdspan and adapting the
// segments to also be mdspans for local access
//
template <distributed_contiguous_range R, std::size_t Rank,
          typename Layout = md::layout_right>
struct mdspan_view : public rng::view_interface<mdspan_view<R, Rank>> {
private:
  using base_type = rng::views::all_t<R>;
  using iterator_type = rng::iterator_t<base_type>;
  using extents_type = md::dextents<std::size_t, Rank>;
  using mdspan_type = md::mdspan<iterator_type, extents_type, Layout,
                                 mdspan_iter_accessor<iterator_type>>;
  using difference_type = rng::iter_difference_t<iterator_type>;

  base_type base_;
  dr::__detail::dr_extents<Rank> full_extents_;
  dr::__detail::dr_extents<Rank> tile_extents_;
  static auto make_segments(auto base, auto tile_extents) {
    auto make_md = [tile_extents](auto v) {
      return mdsegment(std::size_t(std::get<0>(v)), std::get<1>(v),
                       dr::__detail::dr_extents<Rank>{0, 0}, tile_extents);
    };

    return rng::views::enumerate(dr::ranges::segments(base)) |
           rng::views::transform(make_md);
  }
  using segments_type =
      decltype(make_segments(std::declval<base_type>(), tile_extents_));

public:
  mdspan_view(R r, dr::__detail::dr_extents<Rank> full_extents)
      : base_(rng::views::all(std::forward<R>(r))) {
    full_extents_ = full_extents;

    // Default tile extents splits on leading dimension
    tile_extents_ = full_extents;
    tile_extents_[0] = decomp::div;

    replace_decomp();

    segments_ = make_segments(base_, tile_extents_);
  }

  mdspan_view(R r, dr::__detail::dr_extents<Rank> full_extents,
              dr::__detail::dr_extents<Rank> tile_extents)
      : base_(rng::views::all(std::forward<R>(r))), full_extents_(full_extents),
        tile_extents_(tile_extents) {
    replace_decomp();
    segments_ = make_segments(base_, tile_extents_);
  }

  // Base implements random access range
  auto begin() const { return base_.begin(); }
  auto end() const { return base_.end(); }
  auto operator[](difference_type n) { return base_[n]; }

  // Add a local mdspan to the base segment
  // Mdspan access to base
  auto mdspan() const { return mdspan_type(rng::begin(base_), full_extents_); }
  static constexpr auto rank() { return Rank; }

  auto segments() const { return segments_; }

  // Mdspan access to grid
  auto grid() {
    dr::__detail::dr_extents<Rank> grid_extents;
    for (std::size_t i : rng::views::iota(0u, Rank)) {
      grid_extents[i] = full_extents_[i] / tile_extents_[i];
      assert(full_extents_[i] % tile_extents_[i] == 0);
    }
    using grid_iterator_type = rng::iterator_t<segments_type>;
    using grid_type = md::mdspan<grid_iterator_type, extents_type, Layout,
                                 mdspan_iter_accessor<grid_iterator_type>>;
    return grid_type(rng::begin(segments_), grid_extents);
  }

private:
  // Replace div with actual value
  void replace_decomp() {
    auto n = std::size_t(rng::size(dr::ranges::segments(base_)));
    for (std::size_t i = 0; i < Rank; i++) {
      if (tile_extents_[i] == decomp::div) {
        tile_extents_[i] = full_extents_[i] / n;
      } else if (tile_extents_[i] == decomp::all) {
        tile_extents_[i] = full_extents_[i];
      }
      // TODO: Handle this case
      assert(full_extents_[i] % tile_extents_[i] == 0);
    }
  }

  segments_type segments_;
};

template <typename R, std::size_t Rank>
mdspan_view(R &&r, dr::__detail::dr_extents<Rank> extents)
    -> mdspan_view<rng::views::all_t<R>, Rank>;

template <typename R, std::size_t Rank>
mdspan_view(R &&r, dr::__detail::dr_extents<Rank> full_extents,
            dr::__detail::dr_extents<Rank> tile_extents)
    -> mdspan_view<rng::views::all_t<R>, Rank>;

template <typename R>
concept is_mdspan_view =
    rng::viewable_range<R> && requires(R &r) { r.mdspan(); };

} // namespace dr::mhp

namespace dr::mhp::views {

template <std::size_t Rank> class mdspan_adapter_closure {
public:
  mdspan_adapter_closure(dr::__detail::dr_extents<Rank> full_extents,
                         dr::__detail::dr_extents<Rank> tile_extents)
      : full_extents_(full_extents), tile_extents_(tile_extents),
        tile_valid_(true) {}

  mdspan_adapter_closure(dr::__detail::dr_extents<Rank> full_extents)
      : full_extents_(full_extents) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    if (tile_valid_) {
      return mdspan_view(std::forward<R>(r), full_extents_, tile_extents_);
    } else {
      return mdspan_view(std::forward<R>(r), full_extents_);
    }
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const mdspan_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  dr::__detail::dr_extents<Rank> full_extents_;
  dr::__detail::dr_extents<Rank> tile_extents_;
  bool tile_valid_ = false;
};

class mdspan_fn_ {
public:
  template <rng::viewable_range R, typename Extents>
  auto operator()(R &&r, Extents &&full_extents, Extents &&tile_extents) const {
    return mdspan_adapter_closure(std::forward<Extents>(full_extents),
                                  std::forward<Extents>(tile_extents))(
        std::forward<R>(r));
  }

  template <rng::viewable_range R, typename Extents>
  auto operator()(R &&r, Extents &&full_extents) const {
    return mdspan_adapter_closure(std::forward<Extents>(full_extents))(
        std::forward<R>(r));
  }

  template <typename Extents>
  auto operator()(Extents &&full_extents, Extents &&tile_extents) const {
    return mdspan_adapter_closure(std::forward<Extents>(full_extents),
                                  std::forward<Extents>(tile_extents));
  }

  template <typename Extents> auto operator()(Extents &&full_extents) const {
    return mdspan_adapter_closure(std::forward<Extents>(full_extents));
  }
};

inline constexpr auto mdspan = mdspan_fn_{};

} // namespace dr::mhp::views
