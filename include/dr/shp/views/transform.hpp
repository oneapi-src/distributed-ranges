// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/details/ranges_shim.hpp>
#include <dr/shp/normal_distributed_iterator.hpp>

namespace shp {

template <rng::forward_range V, std::copy_constructible F>
class transform_view : public rng::view_interface<transform_view<V, F>> {
public:
  template <rng::viewable_range R>
  transform_view(R &&r, F fn)
      : base_(rng::views::all(r)), fn_(fn), global_view_(base_, fn) {}

  auto begin() const {
    return normal_distributed_iterator<decltype(segments())>(segments(), 0, 0);
  }

  auto end() const {
    auto segs = segments();
    return normal_distributed_iterator<decltype(segments())>(
        segs, size_t(segs.size()), 0);
  }

  auto segments() const {
    return lib::ranges::segments(base_) |
           rng::views::transform([=](auto &&segment) {
             return segment | rng::views::transform(fn_);
           });
  }

private:
  rng::views::all_t<V> base_;
  F fn_;
};

template <rng::viewable_range R, std::copy_constructible F>
transform_view(R &&r, F fn) -> transform_view<rng::views::all_t<R>, F>;

} // namespace shp
