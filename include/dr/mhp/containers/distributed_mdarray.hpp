// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/views/mdspan_view.hpp>

namespace dr::mhp {

template <typename T, std::size_t Rank> class distributed_mdarray {
public:
  using extents_type = dr::__detail::dr_extents<Rank>;

  distributed_mdarray(dr::__detail::dr_extents<Rank> extents,
                      distribution dist = distribution())
      : dv_(md_size(extents), dv_dist(dist, extents)),
        md_view_(make_md_view(dv_, extents)) {}

  auto begin() const { return rng::begin(md_view_); }
  auto end() const { return rng::end(md_view_); }
  auto size() const { return rng::size(md_view_); }
  auto operator[](auto n) { return md_view_[n]; }

  auto segments() { return dr::ranges::segments(dv_); }

  auto mdspan() const { return md_view_.mdspan(); }
  auto grid() { return md_view_.grid(); }
  auto view() const { return md_view_; }

  auto operator==(const distributed_mdarray &other) const {
    return std::equal(begin(), end(), other.begin());
  }

private:
  static auto md_size(auto extents) {
    std::size_t size = 1;
    for (auto extent : extents) {
      size *= extent;
    }
    return size;
  }

  static auto dv_dist(distribution incoming_dist, auto extents) {
    // Granularity matches tile size
    auto tile_extents = extents;
    // TODO: only supports dist on leading dimension
    tile_extents[0] = 1;
    std::size_t tile_size = md_size(tile_extents);
    auto incoming_halo = incoming_dist.halo();
    return distribution().granularity(tile_size).halo(
        incoming_halo.prev * tile_size, incoming_halo.next * tile_size);
  }

  // This wrapper seems to avoid an issue with template argument
  // deduction for mdspan_view
  template <typename DV> static auto make_md_view(DV &&dv, auto extents) {
    return views::mdspan(dv, extents);
  }

  using DV = distributed_vector<T>;
  DV dv_;
  using mdspan_type = decltype(make_md_view(
      std::declval<DV>(), std::declval<dr::__detail::dr_extents<Rank>>()));
  mdspan_type md_view_;
};

template <typename T, std::size_t Rank>
std::ostream &operator<<(std::ostream &os,
                         const distributed_mdarray<T, Rank> &mdarray) {
  os << fmt::format("\n{}", mdarray.mdspan());
  return os;
}

} // namespace dr::mhp
