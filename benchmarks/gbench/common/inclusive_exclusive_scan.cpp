// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

void Inclusive_Exclusive_Scan_DR(benchmark::State &state, bool is_inclusive) {
  xhp::distributed_vector<T> a(default_vector_size, 3);
  xhp::distributed_vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      if (is_inclusive) {
        xhp::inclusive_scan(a, b);
      } else {
        xhp::exclusive_scan(a, b, 0);
      }
    }
  }
}

void Inclusive_Scan_DR(benchmark::State &state) {
  Inclusive_Exclusive_Scan_DR(state, true);
}
DR_BENCHMARK(Inclusive_Scan_DR)

void Exclusive_Scan_DR(benchmark::State &state) {
  Inclusive_Exclusive_Scan_DR(state, false);
}
DR_BENCHMARK(Exclusive_Scan_DR);

#ifdef SYCL_LANGUAGE_VERSION
static void Inclusive_Exclusive_Scan_Reference(benchmark::State &state,
                                               bool is_inclusive) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  auto b = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      if (is_inclusive) {
        std::inclusive_scan(policy, a, a + default_vector_size, b,
                            std::plus<T>{});
      } else {
        std::exclusive_scan(policy, a, a + default_vector_size, b, 0,
                            std::plus<T>{});
      }
    }
  }
  sycl::free(a, q);
  sycl::free(b, q);
}

void Inclusive_Scan_Reference(benchmark::State &state) {
  Inclusive_Exclusive_Scan_Reference(state, true);
}
DR_BENCHMARK(Inclusive_Scan_Reference)

void Exclusive_Scan_Reference(benchmark::State &state) {
  Inclusive_Exclusive_Scan_Reference(state, false);
}
DR_BENCHMARK(Exclusive_Scan_Reference);
#endif

static void Inclusive_Exclusive_Scan_Std(benchmark::State &state,
                                         bool is_inclusive) {
  std::vector<T> a(default_vector_size, 3);
  std::vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      if (is_inclusive) {
        std::inclusive_scan(std::execution::par_unseq, rng::begin(a),
                            rng::end(a), rng::begin(b));
      } else {
        std::exclusive_scan(std::execution::par_unseq, rng::begin(a),
                            rng::end(a), rng::begin(b), 0);
      }
    }
  }
}

void Inclusive_Scan_Std(benchmark::State &state) {
  Inclusive_Exclusive_Scan_Std(state, true);
}
DR_BENCHMARK(Inclusive_Scan_Std)

void Exclusive_Scan_Std(benchmark::State &state) {
  Inclusive_Exclusive_Scan_Std(state, false);
}
DR_BENCHMARK(Exclusive_Scan_Std);
