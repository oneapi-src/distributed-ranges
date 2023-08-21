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

#ifdef SYCL_LANGUAGE_VERSION
static void Sort_DPL(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::sort(policy, a, a + default_vector_size);
    }
  }
  sycl::free(a, q);
}

DR_BENCHMARK(Sort_DPL);
#endif
