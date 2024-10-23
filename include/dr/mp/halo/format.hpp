// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/global.hpp>

#ifdef DR_FORMAT

template <>
struct fmt::formatter<dr::mp::halo_bounds> : formatter<string_view> {
  template <typename FmtContext>
  auto format(dr::mp::halo_bounds hb, FmtContext &ctx) {
    return fmt::format_to(ctx.out(), "prev: {} next: {}", hb.prev, hb.next);
  }
};

#endif
