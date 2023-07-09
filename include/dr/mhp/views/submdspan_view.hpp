// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/views/mdspan_view.hpp>

namespace dr::mhp {

//
// Wrap a mdspan view
//
template <is_mdspan_view Base>
struct submdspan_view : public rng::view_interface<submdspan_view<Base>> {
private:
  using iterator_type = rng::iterator_t<Base>;
  using extents_type = dr::__detail::dr_extents<Base::rank()>;
  using difference_type = rng::iter_difference_t<iterator_type>;

  Base base_;
  extents_type slice_offset_;
  extents_type slice_extents_;

  static auto make_segments(auto base, auto slice_offset, auto slice_extents) {
    return dr::ranges::segments(base);
  }
  using segments_type =
      decltype(make_segments(std::declval<Base>(), std::declval<extents_type>(),
                             std::declval<extents_type>()));

public:
  submdspan_view(is_mdspan_view auto base, extents_type slice_offset,
                 extents_type slice_extents)
      : base_(base), slice_offset_(std::forward<extents_type>(slice_offset)),
        slice_extents_(std::forward<extents_type>(slice_extents)) {
    segments_ = make_segments(base_, slice_offset_, slice_extents_);
  }

  // Base implements random access range
  auto begin() const { return base_.begin(); }
  auto end() const { return base_.end(); }
  auto operator[](difference_type n) { return base_[n]; }

  auto mdspan() const {
    return dr::__detail::make_submdspan(base_.mdspan(), slice_offset_,
                                        slice_extents_);
  }

  auto segments() const { return segments_; }

  // Mdspan access to grid
  auto grid() {
    using grid_iterator_type = rng::iterator_t<segments_type>;
    using grid_type =
        md::mdspan<grid_iterator_type, extents_type, md::layout_right,
                   mdspan_iter_accessor<grid_iterator_type>>;
    return grid_type(rng::begin(segments_), base_.grid().extents());
  }

private:
  segments_type segments_;
};

template <typename R, typename Extents>
submdspan_view(R r, Extents slice_offset, Extents slice_extents)
    -> submdspan_view<R>;

} // namespace dr::mhp

namespace dr::mhp::views {

template <typename Extents> class submdspan_adapter_closure {
public:
  submdspan_adapter_closure(Extents slice_offset, Extents slice_extents)
      : slice_offset_(slice_offset), slice_extents_(slice_extents) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    return submdspan_view(std::forward<R>(r), slice_offset_, slice_extents_);
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const submdspan_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  Extents slice_offset_;
  Extents slice_extents_;
};

class submdspan_fn_ {
public:
  template <is_mdspan_view R, typename Extents>
  auto operator()(R r, Extents &&slice_offset, Extents &&slice_extents) const {
    return submdspan_adapter_closure(std::forward<Extents>(slice_offset),
                                     std::forward<Extents>(slice_extents))(
        std::forward<R>(r));
  }

  template <typename Extents>
  auto operator()(Extents &&slice_offset, Extents &&slice_extents) const {
    return submdspan_adapter_closure(std::forward<Extents>(slice_offset),
                                     std::forward<Extents>(slice_extents));
  }
};

inline constexpr auto submdspan = submdspan_fn_{};

} // namespace dr::mhp::views
