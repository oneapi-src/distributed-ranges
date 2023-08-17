// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/views/mdspan_view.hpp>

namespace dr::mhp::__detail {

//
// Add a local mdspan to the underlying segment
//
template <typename BaseSegment, std::size_t Rank,
          typename Layout = md::layout_stride>
class mdsub_segment : public BaseSegment {
private:
public:
  using index_type = dr::__detail::dr_extents<Rank>;

  mdsub_segment(){};
  mdsub_segment(BaseSegment segment, const index_type &slice_starts,
                const index_type &slice_ends)
      : BaseSegment(segment),
        mdspan_(local_tile(segment, slice_starts, slice_ends)),
        root_mdspan_(segment.mdspan()) {}

  auto mdspan() const { return mdspan_; }
  auto root_mdspan() const { return root_mdspan_; }

private:
  using T = rng::range_value_t<BaseSegment>;

  static auto local_tile(BaseSegment segment, const index_type &slice_starts,
                         const index_type &slice_ends) {
    index_type starts, ends;
    index_type base_starts = segment.origin();
    auto base_mdspan = segment.mdspan();

    for (std::size_t i = 0; i < Rank; i++) {
      // Clip base to area covered by requested span, and translate from global
      // to local indexing
      auto base_end = base_starts[i] + base_mdspan.extent(i);
      starts[i] =
          std::min(base_end, std::max(slice_starts[i], base_starts[i])) -
          base_starts[i];
      ends[i] = std::max(base_starts[i], std::min(slice_ends[i], base_end)) -
                base_starts[i];
    }
    return dr::__detail::make_submdspan(base_mdspan, starts, ends);
  }

  md::mdspan<T, dr::__detail::md_extents<Rank>, md::layout_stride> mdspan_;
  md::mdspan<T, dr::__detail::md_extents<Rank>, md::layout_stride> root_mdspan_;
};

} // namespace dr::mhp::__detail

namespace dr::mhp {

//
// Wrap a mdspan view
//
template <is_mdspan_view Base>
struct submdspan_view : public rng::view_interface<submdspan_view<Base>> {
private:
  static auto make_segments(auto base, auto slice_starts, auto slice_ends) {
    auto make_md = [=](auto segment) {
      return __detail::mdsub_segment(segment, slice_starts, slice_ends);
    };
    return dr::ranges::segments(base) | rng::views::transform(make_md);
  }

  using iterator_type = rng::iterator_t<Base>;
  using extents_type = dr::__detail::dr_extents<Base::rank()>;
  using difference_type = rng::iter_difference_t<iterator_type>;
  using segments_type =
      decltype(make_segments(std::declval<Base>(), std::declval<extents_type>(),
                             std::declval<extents_type>()));

  Base base_;
  extents_type slice_starts_;
  extents_type slice_ends_;
  segments_type segments_;

public:
  submdspan_view(is_mdspan_view auto base, extents_type slice_starts,
                 extents_type slice_ends)
      : base_(base), slice_starts_(std::forward<extents_type>(slice_starts)),
        slice_ends_(std::forward<extents_type>(slice_ends)) {
    segments_ = make_segments(base_, slice_starts_, slice_ends_);
  }

  // Base implements random access range
  auto begin() const { return base_.begin(); }
  auto end() const { return base_.end(); }
  auto operator[](difference_type n) { return base_[n]; }

  auto mdspan() const {
    return dr::__detail::make_submdspan(base_.mdspan(), slice_starts_,
                                        slice_ends_);
  }

  auto segments() const { return segments_; }

  // Mdspan access to grid
  auto grid() {
    using grid_iterator_type = rng::iterator_t<segments_type>;
    using grid_type =
        md::mdspan<grid_iterator_type, dr::__detail::md_extents<Base::rank()>,
                   md::layout_right,
                   dr::__detail::mdspan_iter_accessor<grid_iterator_type>>;
    return grid_type(rng::begin(segments_), base_.grid().extents());
  }
};

template <typename R, typename Extents>
submdspan_view(R r, Extents slice_starts, Extents slice_ends)
    -> submdspan_view<R>;

} // namespace dr::mhp

namespace dr::mhp::views {

template <typename Extents> class submdspan_adapter_closure {
public:
  submdspan_adapter_closure(Extents slice_starts, Extents slice_ends)
      : slice_starts_(slice_starts), slice_ends_(slice_ends) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    return submdspan_view(std::forward<R>(r), slice_starts_, slice_ends_);
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const submdspan_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  Extents slice_starts_;
  Extents slice_ends_;
};

class submdspan_fn_ {
public:
  template <is_mdspan_view R, typename Extents>
  auto operator()(R r, Extents &&slice_starts, Extents &&slice_ends) const {
    return submdspan_adapter_closure(std::forward<Extents>(slice_starts),
                                     std::forward<Extents>(slice_ends))(
        std::forward<R>(r));
  }

  template <typename Extents>
  auto operator()(Extents &&slice_starts, Extents &&slice_ends) const {
    return submdspan_adapter_closure(std::forward<Extents>(slice_starts),
                                     std::forward<Extents>(slice_ends));
  }
};

inline constexpr auto submdspan = submdspan_fn_{};

} // namespace dr::mhp::views
