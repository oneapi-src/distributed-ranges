// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/mp/views/mdspan_view.hpp>

namespace dr::mp {

template <typename T, std::size_t Rank> class distributed_mdarray {
public:
  using shape_type = dr::__detail::dr_extents<Rank>;
  static constexpr auto rank() { return Rank; }

  distributed_mdarray(dr::__detail::dr_extents<Rank> shape,
                      distribution dist = distribution())
      : tile_shape_(tile_shape(shape)), dv_(dv_size(), dv_dist(dist, shape)),
        md_view_(make_md_view(dv_, shape, tile_shape_)) {}

  auto begin() const { return rng::begin(md_view_); }
  auto end() const { return rng::end(md_view_); }
  auto size() const { return rng::size(md_view_); }
  auto operator[](auto n) { return md_view_[n]; }

  auto segments() { return dr::ranges::segments(md_view_); }
  auto &halo() const { return dr::mp::halo(dv_); }

  auto mdspan() const { return md_view_.mdspan(); }
  auto extent(std::size_t r) const { return mdspan().extent(r); }
  auto grid() { return md_view_.grid(); }
  auto view() const { return md_view_; }

  auto operator==(const distributed_mdarray &other) const {
    return std::equal(begin(), end(), other.begin());
  }

private:
  using DV = distributed_vector<T>;

  static auto tile_shape(auto shape) {
    std::size_t n = default_comm().size(); // dr-style ignore
    shape[0] = dr::__detail::partition_up(shape[0], n);
    return shape;
  }

  static auto md_size(auto shape) {
    std::size_t size = 1;
    for (auto extent : shape) {
      size *= extent;
    }
    return size;
  }

  auto dv_size() {
    return default_comm().size() * md_size(tile_shape_); // dr-style ignore
  }

  static auto dv_dist(distribution incoming_dist, auto shape) {
    // Decomp is 1 "row" in decomp dimension
    // TODO: only supports dist on leading dimension
    shape[0] = 1;
    std::size_t row_size = md_size(shape);
    auto incoming_halo = incoming_dist.halo();
    return distribution().halo(incoming_halo.prev * row_size,
                               incoming_halo.next * row_size);
  }

  // This wrapper seems to avoid an issue with template argument
  // deduction for mdspan_view
  static auto make_md_view(const DV &dv, shape_type shape,
                           shape_type tile_shape) {
    return views::mdspan(dv, shape, tile_shape);
  }

  shape_type tile_shape_;
  DV dv_;
  using mdspan_type =
      decltype(make_md_view(std::declval<DV>(), std::declval<shape_type>(),
                            std::declval<shape_type>()));
  mdspan_type md_view_;
};

template <typename T, std::size_t Rank>
auto &halo(const distributed_mdarray<T, Rank> &mdarray) {
  return mdarray.halo();
}

template <typename T, std::size_t Rank>
std::ostream &operator<<(std::ostream &os,
                         const distributed_mdarray<T, Rank> &mdarray) {
  os << fmt::format("\n{}", mdarray.mdspan());
  return os;
}

} // namespace dr::mp
