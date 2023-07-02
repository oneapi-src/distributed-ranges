// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/views/mdspan_vew.hpp>

namespace dr::mhp {

template <typename T, std::size_t Rank, typename Layout = md::layout_right>
class mdarray {
public:
  mdarray(dr_extents<Rank> extents)
      : dv_(dv_size(extents)), md_view_(dv_, extents){};

private:
  static auto dv_size(auto extents) {
    return std::accumulate(extents.begin(), extents.end(),
                           std::multiples<std::size_t>());
  }

  using dv_type = distributed_vector<T>;
  dv_type dv_;
  mdspan_view<dv_type, Rank, Layout> md_view_;
};

} // namespace dr::mhp
