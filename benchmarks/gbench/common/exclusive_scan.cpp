// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

static void Exclusive_Scan_DR(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size, 3);
  xhp::distributed_vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::exclusive_scan(a, b, 0);
    }
  }
}

DR_BENCHMARK(Exclusive_Scan_DR);

#ifdef SYCL_LANGUAGE_VERSION
static void Exclusive_Scan_stdplus_EXP(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  auto b = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::exclusive_scan(policy, a, a + default_vector_size, b, 0, std::plus<>{});
    }
  }
  sycl::free(a, q);
  sycl::free(b, q);
}

DR_BENCHMARK(Exclusive_Scan_stdplus_EXP);

static void Exclusive_Scan_stdplusT_EXP(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  auto b = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::exclusive_scan(policy, a, a + default_vector_size, b, 0,
                          std::plus<T>{});
    }
  }
  sycl::free(a, q);
  sycl::free(b, q);
}

DR_BENCHMARK(Exclusive_Scan_stdplusT_EXP);

static void Exclusive_Scan_none_EXP(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  auto b = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::exclusive_scan(policy, a, a + default_vector_size, b, 0);
    }
  }
  sycl::free(a, q);
  sycl::free(b, q);
}

DR_BENCHMARK(Exclusive_Scan_none_EXP);

static void Exclusive_Scan_Reference(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  auto b = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::exclusive_scan(policy, a, a + default_vector_size, b, 0,
                          std::plus<T>{});
    }
  }
  sycl::free(a, q);
  sycl::free(b, q);
}

DR_BENCHMARK(Exclusive_Scan_Reference);
#endif

static void Exclusive_Scan_Std(benchmark::State &state) {
  std::vector<T> a(default_vector_size, 3);
  std::vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::exclusive_scan(std::execution::par_unseq, rng::begin(a), rng::end(a),
                          rng::begin(b), 0);
    }
  }
}

DR_BENCHMARK(Exclusive_Scan_Std);
