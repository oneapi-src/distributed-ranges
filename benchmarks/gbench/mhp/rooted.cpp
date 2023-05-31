// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-bench.hpp"

using T = double;

static void CopyDist2Local_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  std::vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::copy(0, src, dst.begin());
    }
  }
  memory_bandwidth(state,
                   2 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(CopyDist2Local_DR);

static void CopyLocal2Dist_DR(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  xhp::distributed_vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::copy(0, src, dst.begin());
    }
  }
  memory_bandwidth(state,
                   2 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(CopyLocal2Dist_DR);
