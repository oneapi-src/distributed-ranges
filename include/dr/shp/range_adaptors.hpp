// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <shp/views/standard_views.hpp>
#include <shp/zip_view.hpp>

namespace shp {

template <rng::range R> auto enumerate(R &&r) {
  auto i = rng::views::iota(uint32_t(0), uint32_t(rng::size(r)));
  return shp::zip_view(i, r);
}

} // namespace shp
