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
      : tile_extents_(tile_extents(extents)),
        dv_(dv_size(), dv_dist(dist, extents)),
        md_view_(make_md_view(dv_, extents, tile_extents_)) {}

  auto begin() const { return rng::begin(md_view_); }
  auto end() const { return rng::end(md_view_); }
  auto size() const { return rng::size(md_view_); }
  auto operator[](auto n) { return md_view_[n]; }

  auto segments() { return dr::ranges::segments(md_view_); }
  auto &halo() const { return dr::mhp::halo(dv_); }

  auto mdspan() const { return md_view_.mdspan(); }
  auto grid() { return md_view_.grid(); }
  auto view() const { return md_view_; }

  auto operator==(const distributed_mdarray &other) const {
    return std::equal(begin(), end(), other.begin());
  }

private:
  using DV = distributed_vector<T>;

  static auto tile_extents(auto extents) {
    extents[0] = dr::__detail::round_up(
        extents[0], default_comm().size()); // dr-style ignore
    return extents;
  }

  static auto md_size(auto extents) {
    std::size_t size = 1;
    for (auto extent : extents) {
      size *= extent;
    }
    return size;
  }

  auto dv_size() {
    return default_comm().size() * md_size(tile_extents_); // dr-style ignore
  }

  static auto dv_dist(distribution incoming_dist, auto extents) {
    // Decomp is 1 "row" in decomp dimension
    // TODO: only supports dist on leading dimension
    extents[0] = 1;
    std::size_t row_size = md_size(extents);
    auto incoming_halo = incoming_dist.halo();
    return distribution().halo(incoming_halo.prev * row_size,
                               incoming_halo.next * row_size);
  }

  // This wrapper seems to avoid an issue with template argument
  // deduction for mdspan_view
  static auto make_md_view(const DV &dv, extents_type extents,
                           extents_type tile_extents) {
    return views::mdspan(dv, extents, tile_extents);
  }

  extents_type tile_extents_;
  DV dv_;
  using mdspan_type =
      decltype(make_md_view(std::declval<DV>(), std::declval<extents_type>(),
                            std::declval<extents_type>()));
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

} // namespace dr::mhp
