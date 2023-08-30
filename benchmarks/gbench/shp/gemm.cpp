// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

#include <dr/shp.hpp>

#include <oneapi/mkl.hpp>

template <rng::forward_range X> void fill_random(X &&x) {
  for (auto &&value : x) {
    value = drand48() * 100;
  }
}

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

static void Gemm_Reference(benchmark::State &state) {
  auto q = get_queue();

  std::size_t m = 32;
  std::size_t n = 32;
  std::size_t k = 32;

  std::vector<T> a_local(m * k);
  std::vector<T> b_local(k * n);
  std::vector<T> c_local(m * n);

  fill_random(a_local);
  fill_random(b_local);
  fill_random(c_local);

  T *a = sycl::malloc_device<T>(m * k, q);
  T *b = sycl::malloc_device<T>(k * n, q);
  T *c = sycl::malloc_device<T>(m * n, q);

  q.memcpy(a, a_local.data(), m * k * sizeof(T)).wait();
  q.memcpy(b, b_local.data(), k * n * sizeof(T)).wait();
  q.memcpy(c, c_local.data(), m * n * sizeof(T)).wait();

  Stats stats(state, (m * k + k * n) * sizeof(T), m * n * sizeof(T));

  for (auto _ : state) {
    stats.rep();
    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans,
                                       oneapi::mkl::transpose::nontrans, m, n,
                                       k, T(1), a, m, b, n, T(1), c, k);
  }
}

DR_BENCHMARK(Gemm_Reference);
