// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ranges>
#include <shp/views/standard_views.hpp>
#include <shp/zip_view.hpp>

namespace shp {

template <std::ranges::range R> auto enumerate(R &&r) {
  auto i =
      std::ranges::views::iota(uint32_t(0), uint32_t(std::ranges::size(r)));
  return shp::zip_view(i, r);
}

} // namespace shp
