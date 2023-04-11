// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/details/segments_tools.hpp>
#include <dr/shp/containers/index.hpp>
#include <dr/shp/distributed_span.hpp>
#include <dr/shp/views/enumerate.hpp>
#include <dr/shp/zip_view.hpp>
#include <dr/views/transform.hpp>

namespace shp {

namespace views {

template <lib::distributed_range R>
auto slice(R &&r, shp::index<> slice_indices) {
  return shp::distributed_span(lib::ranges::segments(std::forward<R>(r)))
      .subspan(slice_indices[0], slice_indices[1] - slice_indices[0]);
}

class slice_adaptor_closure {
public:
  slice_adaptor_closure(shp::index<> slice_indices) : idx_(slice_indices) {}

  template <rng::random_access_range R> auto operator()(R &&r) const {
    return slice(std::forward<R>(r), idx_);
  }

  template <rng::random_access_range R>
  friend auto operator|(R &&r, const slice_adaptor_closure &closure) {
    return closure(std::forward<R>(r));
  }

private:
  shp::index<> idx_;
};

inline auto slice(shp::index<> slice_indices) {
  return slice_adaptor_closure(slice_indices);
}

} // namespace views

} // namespace shp
