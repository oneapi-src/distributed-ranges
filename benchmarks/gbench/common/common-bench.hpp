// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
#endif

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

#ifdef SYCL_LANGUAGE_VERSION

inline void show_device(sycl::device device) {
  fmt::print("  name: {}\n"
             "    max compute units: {}\n",
             device.get_info<sycl::info::device::name>(),
             device.get_info<sycl::info::device::max_compute_units>());
}

#endif
