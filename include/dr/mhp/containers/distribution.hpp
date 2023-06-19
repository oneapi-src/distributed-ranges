// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/halo.hpp>

namespace dr::mhp {

struct distribution {
public:
  distribution &halo(std::size_t radius) {
    halo_bounds_.prev = radius;
    halo_bounds_.next = radius;
    return *this;
  }

  distribution &halo(std::size_t prev, std::size_t next) {
    halo_bounds_.prev = prev;
    halo_bounds_.next = next;
    return *this;
  }

  auto halo() const { return halo_bounds_; }

  distribution &periodic(bool periodic) {
    halo_bounds_.periodic = periodic;
    return *this;
  }

  auto periodic() const { return halo_bounds_.periodic; }

  distribution &granularity(std::size_t size) {
    granularity_ = size;
    return *this;
  }

  auto granularity() const { return granularity_; }

private:
  halo_bounds halo_bounds_;
  std::size_t granularity_ = 1;
};

} // namespace dr::mhp
