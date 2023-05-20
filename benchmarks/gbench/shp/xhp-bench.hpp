// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include "dr/shp.hpp"
#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace xhp = dr::shp;

extern std::size_t default_vector_size;
extern std::size_t default_repetitions;

inline std::size_t comm_rank = 0;
inline std::size_t comm_size = 1;

#define BENCH_SHP
