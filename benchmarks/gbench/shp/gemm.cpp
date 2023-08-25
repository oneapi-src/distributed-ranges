// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

#include <oneapi/mkl.hpp>

static void Gemm_DR(benchmark::State &state) {
  auto q = get_queue();

  std::size_t m = 32;
  std::size_t n = 32;
  std::size_t k = 32;

  auto partitions = dr::shp::partition_matmul(m, n, k);
  dr::shp::distributed_dense_matrix<T> a({m, k}, partitions[0]);
  dr::shp::distributed_dense_matrix<T> b({k, n}, partitions[1]);
  dr::shp::distributed_dense_matrix<T> result({m, n}, partitions[2]);

  Stats stats(state, (m * k + k * n) * sizeof(T), m * n * sizeof(T));
  a[{2, 3}] = 12;
  a[{5, 7}] = 42;
  a[{8, 9}] = 37;

  b[{2, 3}] = 12;
  b[{5, 7}] = 42;
  b[{8, 9}] = 37;

  for (auto _ : state) {
    stats.rep();
    dr::shp::gemm_inplace(a, b, result);
  }
}

DR_BENCHMARK(Gemm_DR);
