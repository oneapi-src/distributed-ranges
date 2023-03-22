// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include "dr/shp/shp.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

extern std::size_t comm_rank;
extern std::size_t comm_size;

// Namespace aliases and wrapper functions to make the tests uniform
namespace zhp = shp::views;
namespace xhp = shp;

inline void barrier() {}
inline void fence() {}
inline auto iota(auto &&r, const auto &val) {
  return std::iota(r.begin(), r.end(), val);
}

template <typename T, typename Alloc>
inline auto default_policy(const shp::distributed_vector<T, Alloc> &dv) {
  return shp::par_unseq;
}
using AllocatorTypes = ::testing::Types<shp::device_allocator<int>>;

#include "common-tests.hpp"
