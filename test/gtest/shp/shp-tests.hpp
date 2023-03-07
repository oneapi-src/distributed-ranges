// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include "dr/shp/shp.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

extern int comm_rank;
extern int comm_size;

namespace zhp = shp::views;
namespace xhp = shp;

inline void barrier() {}
inline void fence() {}

using AllocatorTypes =
    ::testing::Types<shp::device_allocator<int>,
                     shp::shared_allocator<long long unsigned int>>;

#include "common-tests.hpp"
