// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/sp/views/standard_views.hpp>
#include <dr/sp/zip_view.hpp>

namespace dr::sp {

template <rng::range R> auto enumerate(R &&r) {
  auto i = rng::views::iota(uint32_t(0), uint32_t(rng::size(r)));
  return dr::sp::zip_view(i, r);
}

} // namespace dr::sp
