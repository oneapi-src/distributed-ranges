// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <vendor/source_location/source_location.hpp>

extern std::size_t comm_rank;
extern std::size_t comm_size;

extern std::size_t default_vector_size;
extern std::size_t default_repetitions;

inline void memory_bandwidth(benchmark::State &state, std::size_t bytes) {
  state.counters["Memory"] =
      benchmark::Counter(bytes, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1024);
}
