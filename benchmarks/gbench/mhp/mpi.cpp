// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-bench.hpp"
static void Barrier(benchmark::State &state) {
  for (auto _ : state) {
    dr::mhp::barrier();
  }
}

BENCHMARK(Barrier)->Iterations(1000000);

static void Fence(benchmark::State &state) {
  for (auto _ : state) {
    dr::mhp::fence();
  }
}

BENCHMARK(Fence)->Iterations(100000000);
