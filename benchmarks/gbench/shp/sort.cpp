// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

static void Sort_DR(benchmark::State &state) {
  dr::shp::distributed_vector<T> a(default_vector_size, 3);
  Stats stats(state, sizeof(T) * a.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      dr::shp::sort(a);
    }
  }
}

DR_BENCHMARK(Sort_DR);
