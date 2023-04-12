// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/shp/views/standard_views.hpp>
#include <dr/views/views.hpp>

namespace dr::shp {

namespace views {

inline constexpr auto transform = dr::views::transform;

inline constexpr auto take = rng::views::take;

inline constexpr auto drop = rng::views::drop;

} // namespace views

} // namespace dr::shp
