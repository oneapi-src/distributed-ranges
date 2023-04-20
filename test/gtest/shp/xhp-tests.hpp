// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include "dr/shp.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

// To share tests with MHP
const std::size_t comm_rank = 0;
const std::size_t comm_size = 1;

// Namespace aliases and wrapper functions to make the tests uniform
namespace xhp = dr::shp;

inline void barrier() {}
inline void fence() {}

using AllocatorTypes = ::testing::Types<dr::shp::device_allocator<int>>;

template <typename V>
concept compliant_view = rng::forward_range<V> && requires(V &v) {
  dr::ranges::segments(v);
  dr::ranges::rank(dr::ranges::segments(v)[0]);
};

#include "common-tests.hpp"

using AllTypes = ::testing::Types<xhp::distributed_vector<int>,
                                  xhp::distributed_vector<float>>;
