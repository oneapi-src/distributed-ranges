// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

static void Barrier(benchmark::State &state) {
  for (auto _ : state) {
    dr::mp::barrier();
  }
}

DR_BENCHMARK_BASE(Barrier)->Iterations(1000000);

static void Fence(benchmark::State &state) {
  for (auto _ : state) {
    dr::mp::fence();
  }
}

DR_BENCHMARK_BASE(Fence)->Iterations(100000000);
