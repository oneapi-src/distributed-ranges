// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vendor/source_location/source_location.hpp>

#include "dr/mhp.hpp"

namespace xhp = dr::mhp;

extern std::size_t default_vector_size;
extern std::size_t default_repetitions;
extern std::size_t stencil_steps;

#define BENCH_MHP
