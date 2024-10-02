// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mp/halo.hpp>

namespace dr::mp {

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

  auto halo() const {
    halo_bounds halo_bounds_resized = halo_bounds_;
    halo_bounds_resized.prev *= redundancy_;
    halo_bounds_resized.next *= redundancy_;
    return halo_bounds_resized;
  }

  distribution &redundancy(std::size_t redundancy) {
    redundancy_ = redundancy;
    return *this;
  }

  auto redundancy() const { return redundancy_; }

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
  std::size_t redundancy_ = 1;
  std::size_t granularity_ = 1;
};

struct extended_local_data_distribution {
  std::size_t begin;
  std::size_t end;
  std::size_t segment_size;

  extended_local_data_distribution() = default;
  extended_local_data_distribution(std::size_t segment_size,
                                   std::size_t size,
                                   halo_bounds hb)
      : segment_size(segment_size) {
    if (default_comm().rank() * segment_size >= hb.prev)
      begin = default_comm().rank() * segment_size - hb.prev;
    else
      begin = 0;
    end = std::min((default_comm().rank() + 1) * segment_size + hb.next, size);
  }
};

} // namespace dr::mp
