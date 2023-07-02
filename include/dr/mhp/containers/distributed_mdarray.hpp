// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/views/mdspan_view.hpp>

namespace dr::mhp {

template <typename T, std::size_t Rank, typename Layout = md::layout_right>
class distributed_mdarray {
public:
  distributed_mdarray(dr_extents<Rank> extents)
      : dv_(dv_size(extents)), md_view_(make_md_view(dv_, extents)) {}

  auto begin() const { return rng::begin(md_view_); }
  auto end() const { return rng::end(md_view_); }
  auto size() const { return rng::size(md_view_); }
  auto operator[](auto n) { return md_view_[n]; }

  auto segments() { return dr::ranges::segments(dv_); }

private:
  static auto dv_size(auto extents) {
    std::size_t size = 1;
    for (auto extent : extents) {
      size *= extent;
    }
    return size;
  }

  template <typename DV> static auto make_md_view(DV &&dv, auto extents) {
    // return views::mdspan(rng::views::all(dv), extents);
    return views::mdspan(dv, extents);
  }

  using DV = distributed_vector<T>;
  DV dv_;
  using mdspan_type = decltype(make_md_view(std::declval<DV>(),
                                            std::declval<dr_extents<Rank>>()));
  mdspan_type md_view_;
};

} // namespace dr::mhp
