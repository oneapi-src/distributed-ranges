// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/halo.hpp>

namespace dr::mhp {

struct distribution {
public:
  distribution &halo(std::size_t radius) {
    halo_bounds_ = halo_bounds(radius);
    return *this;
  }

  auto halo() { return halo_bounds_; }

private:
  halo_bounds halo_bounds_;
};

} // namespace dr::mhp
