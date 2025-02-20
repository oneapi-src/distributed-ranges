// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/mdspan_shim.hpp>
#include <dr/detail/mdspan_utils.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/detail/ranges_utils.hpp>
#include <dr/mp/containers/distributed_vector.hpp>

namespace dr::mp::decomp {

inline constexpr std::size_t div = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t all = div - 1;

} // namespace dr::mp::decomp

namespace dr::mp::__detail {

//
// Add a local mdspan to the underlying segment
//
template <typename BaseSegment, std::size_t Rank>
class md_segment : public rng::view_interface<md_segment<BaseSegment, Rank>> {
private:
  using stencil_index_type = dr::__detail::dr_extents<Rank>;

public:
  using index_type = dr::__detail::dr_extents<Rank>;

  md_segment() {}
  md_segment(index_type origin, BaseSegment segment, index_type tile_shape,
             extended_local_data_distribution ext_dist)
      : base_(segment), origin_(origin),
        mdspan_(local_tile(segment, tile_shape)),
        mdspan_extended_(local_tile_extended(segment, tile_shape)),
        ext_dist_(ext_dist) {
    dr::drlog.debug(dr::logger::mdspan_view,
                    "md_segment\n  origin: {}\n  tile shape: {}\n", origin,
                    tile_shape);

    for (std::size_t i = 0; i < Rank; i++)
      end_[i] = origin_[i] + tile_shape[i];
  }

  // view_interface uses below to define everything else
  auto begin() const { return base_.begin(); }
  auto end() const { return base_.end(); }

  auto reserved() const { return base_.reserved(); }

  auto halo() const { return dr::mp::halo(base_); }

  [[nodiscard]] stencil_index_type
  begin_stencil(stencil_index_type stencil) const {
    stencil_index_type out;
    // Supports only 1d distribution
    for (std::size_t i = 0; i < Rank; i++) {
      out[i] = std::min(std::max(origin_[i],
                                 (i == 0 ? ext_dist_.begin : origin_[i]) +
                                     stencil[i]),
                        end_[i]) -
               origin_[i];
    }
    return out;
  }
  [[nodiscard]] stencil_index_type
  end_stencil(stencil_index_type stencil) const {
    stencil_index_type out;
    // Supports only 1d distribution
    for (std::size_t i = 0; i < Rank; i++) {
      out[i] = std::max(std::min(end_[i], (i == 0 ? ext_dist_.end : end_[i]) -
                                              stencil[i]),
                        origin_[i]) -
               origin_[i];
    }
    return out;
  }
  [[nodiscard]] std::pair<stencil_index_type, stencil_index_type>
  stencil(stencil_index_type begin, stencil_index_type end) const {
    return {begin_stencil(begin), end_stencil(end)};
  }

  // mdspan-specific methods
  auto mdspan() const { return mdspan_; }
  auto mdspan_extended() const { return mdspan_extended_; }
  auto origin() const { return origin_; }
  // for slices, this would be the underlying mdspan
  auto root_mdspan() const { return mdspan(); }

private:
  using T = rng::range_value_t<BaseSegment>;

  static auto local_tile(BaseSegment segment, const index_type &tile_shape) {
    // Undefined behavior if the segments is not local
    T *ptr = dr::ranges::rank(segment) == default_comm().rank()
                 ? std::to_address(dr::ranges::local(rng::begin(segment)))
                 : nullptr;
    return md::mdspan(ptr, tile_shape);
  }

  static auto local_tile_extended(BaseSegment segment,
                                  const index_type &tile_shape) {
    T *ptr = std::to_address(dr::ranges::local(rng::begin(segment)));
    return md::mdspan(ptr, tile_shape);
  }

  BaseSegment base_;
  index_type origin_;
  index_type end_;
  md::mdspan<T, dr::__detail::md_extents<Rank>, md::layout_stride> mdspan_;
  md::mdspan<T, dr::__detail::md_extents<Rank>, md::layout_stride>
      mdspan_extended_;

  extended_local_data_distribution ext_dist_;
};

} // namespace dr::mp::__detail

namespace dr::mp {

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
  using mdspan_type =
      md::mdspan<iterator_type, extents_type, Layout,
                 dr::__detail::mdspan_iter_accessor<iterator_type>>;
  using difference_type = rng::iter_difference_t<iterator_type>;
  using index_type = dr::__detail::dr_extents<Rank>;

  base_type base_;
  index_type full_shape_;
  index_type tile_shape_;

  static auto segment_index_to_global_origin(std::size_t linear,
                                             index_type full_shape,
                                             index_type tile_shape) {
    index_type grid_shape;
    for (std::size_t i = 0; i < Rank; i++) {
      grid_shape[i] = dr::__detail::partition_up(full_shape[i], tile_shape[i]);
    }
    auto origin = dr::__detail::linear_to_index(linear, grid_shape);
    for (std::size_t i = 0; i < Rank; i++) {
      origin[i] *= tile_shape[i];
    }

    return origin;
  }

  static auto make_segments(auto base, auto full_shape, auto tile_shape,
                            auto dist) {
    extended_local_data_distribution ext_dist(tile_shape[0], full_shape[0],
                                              dist.halo());

    auto make_md = [=](auto v) {
      auto clipped = tile_shape;
      std::size_t segment_index = std::get<0>(v);
      std::size_t end = (segment_index + 1) * tile_shape[0];
      if (end > full_shape[0]) {
        clipped[0] -= std::min(end - full_shape[0], clipped[0]);
      }
      return __detail::md_segment(
          segment_index_to_global_origin(segment_index, full_shape, tile_shape),
          std::get<1>(v), clipped, ext_dist);
    };

    dr::drlog.debug(dr::logger::mdspan_view,
                    "mdspan_view\n  full shape: {}\n  tile shape: {}\n",
                    full_shape, tile_shape);
    // use bounded_enumerate so we get a std::ranges::common_range
    return dr::__detail::bounded_enumerate(dr::ranges::segments(base)) |
           rng::views::transform(make_md);
  }
  using segments_type =
      decltype(make_segments(std::declval<base_type>(), full_shape_,
                             tile_shape_, std::declval<distribution>()));

public:
  mdspan_view(R r, dr::__detail::dr_extents<Rank> full_shape, distribution dist)
      : base_(rng::views::all(std::forward<R>(r))) {
    full_shape_ = full_shape;

    // Default tile shape splits on leading dimension
    tile_shape_ = full_shape;
    tile_shape_[0] = decomp::div;

    replace_decomp();
    segments_ = make_segments(base_, full_shape_, tile_shape_, dist);
  }

  mdspan_view(R r, dr::__detail::dr_extents<Rank> full_shape,
              dr::__detail::dr_extents<Rank> tile_shape, distribution dist)
      : base_(rng::views::all(std::forward<R>(r))), full_shape_(full_shape),
        tile_shape_(tile_shape) {
    replace_decomp();
    segments_ = make_segments(base_, full_shape_, tile_shape_, dist);
  }

  // Base implements random access range
  auto begin() const { return base_.begin(); }
  auto end() const { return base_.end(); }
  auto operator[](difference_type n) { return base_[n]; }

  // Add a local mdspan to the base segment
  // Mdspan access to base
  auto mdspan() const { return mdspan_type(rng::begin(base_), full_shape_); }
  static constexpr auto rank() { return Rank; }

  auto segments() const { return segments_; }

  // Mdspan access to grid
  auto grid() {
    dr::__detail::dr_extents<Rank> grid_shape;
    for (std::size_t i : rng::views::iota(0u, Rank)) {
      grid_shape[i] =
          dr::__detail::partition_up(full_shape_[i], tile_shape_[i]);
    }
    using grid_iterator_type = rng::iterator_t<segments_type>;
    using grid_type =
        md::mdspan<grid_iterator_type, extents_type, Layout,
                   dr::__detail::mdspan_iter_accessor<grid_iterator_type>>;
    return grid_type(rng::begin(segments_), grid_shape);
  }

private:
  // Replace div with actual value
  void replace_decomp() {
    auto n = std::size_t(rng::size(dr::ranges::segments(base_)));
    for (std::size_t i = 0; i < Rank; i++) {
      if (tile_shape_[i] == decomp::div) {
        tile_shape_[i] = dr::__detail::partition_up(full_shape_[i], n);
      } else if (tile_shape_[i] == decomp::all) {
        tile_shape_[i] = full_shape_[i];
      }
    }
  }

  segments_type segments_;
};

template <typename R, std::size_t Rank>
mdspan_view(R &&r, dr::__detail::dr_extents<Rank> extents, distribution dist)
    -> mdspan_view<rng::views::all_t<R>, Rank>;

template <typename R, std::size_t Rank>
mdspan_view(R &&r, dr::__detail::dr_extents<Rank> full_shape,
            dr::__detail::dr_extents<Rank> tile_shape, distribution dist)
    -> mdspan_view<rng::views::all_t<R>, Rank>;

template <typename R>
concept is_mdspan_view = dr::distributed_range<R> && requires(R &r) {
  r.mdspan();
};

} // namespace dr::mp

namespace dr::mp::views {

template <std::size_t Rank> class mdspan_adapter_closure {
public:
  mdspan_adapter_closure(dr::__detail::dr_extents<Rank> full_shape,
                         dr::__detail::dr_extents<Rank> tile_shape,
                         distribution dist)
      : full_shape_(full_shape), tile_shape_(tile_shape), tile_valid_(true),
        dist_(dist) {}

  mdspan_adapter_closure(dr::__detail::dr_extents<Rank> full_shape,
                         distribution dist)
      : full_shape_(full_shape), dist_(dist) {}

  template <rng::viewable_range R> auto operator()(R &&r) const {
    if (tile_valid_) {
      return mdspan_view(std::forward<R>(r), full_shape_, tile_shape_, dist_);
    } else {
      return mdspan_view(std::forward<R>(r), full_shape_, dist_);
    }
  }

  template <rng::viewable_range R>
  friend auto operator|(R &&r, const mdspan_adapter_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  dr::__detail::dr_extents<Rank> full_shape_;
  dr::__detail::dr_extents<Rank> tile_shape_;
  bool tile_valid_ = false;
  distribution dist_;
};

class mdspan_fn_ {
public:
  template <rng::viewable_range R, typename Shape>
  auto operator()(R &&r, Shape &&full_shape, Shape &&tile_shape,
                  distribution dist) const {
    return mdspan_adapter_closure(std::forward<Shape>(full_shape),
                                  std::forward<Shape>(tile_shape),
                                  dist)(std::forward<R>(r));
  }

  template <rng::viewable_range R, typename Shape>
  auto operator()(R &&r, Shape &&full_shape, distribution dist) const {
    return mdspan_adapter_closure(std::forward<Shape>(full_shape),
                                  dist)(std::forward<R>(r));
  }

  template <typename Shape>
  auto operator()(Shape &&full_shape, Shape &&tile_shape,
                  distribution dist) const {
    return mdspan_adapter_closure(std::forward<Shape>(full_shape),
                                  std::forward<Shape>(tile_shape), dist);
  }

  template <typename Shape>
  auto operator()(Shape &&full_shape, distribution dist) const {
    return mdspan_adapter_closure(std::forward<Shape>(full_shape), dist);
  }
};

inline constexpr auto mdspan = mdspan_fn_{};

} // namespace dr::mp::views
