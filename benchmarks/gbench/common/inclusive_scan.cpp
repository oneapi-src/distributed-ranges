// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = double;

static void Inclusive_Scan(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size, 3);
  xhp::distributed_vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::inclusive_scan(a, b);
    }
  }
}

DR_BENCHMARK(Inclusive_Scan);

static void Inclusive_Scan_Serial(benchmark::State &state) {
  std::vector<T> a(default_vector_size, 3);
  std::vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::inclusive_scan(std::execution::par_unseq, rng::begin(a), rng::end(a),
                          rng::begin(b));
    }
  }
}

DR_BENCHMARK(Inclusive_Scan_Serial);
