// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = float;

template <rng::forward_range X> void fill_random(X &&x) {
  for (auto &&value : x) {
    value = drand48() * 100;
  }
}

static void Sort_DR(benchmark::State &state) {
  dr::shp::distributed_vector<T> a(default_vector_size);
  fill_random(a);
  Stats stats(state, sizeof(T) * a.size());
  for (auto _ : state) {
    stats.rep();
    dr::shp::sort(a);
  }
}

DR_BENCHMARK(Sort_DR);

#ifdef SYCL_LANGUAGE_VERSION
static void Sort_DPL(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  std::vector<T> a_local(default_vector_size);
  fill_random(a_local);
  auto a = sycl::malloc_device<T>(default_vector_size, q);
  q.memcpy(a, a_local.data(), default_vector_size * sizeof(T)).wait();
  Stats stats(state, sizeof(T) * default_vector_size);

  for (auto _ : state) {
    stats.rep();
    std::sort(policy, a, a + default_vector_size);
  }
  sycl::free(a, q);
}

DR_BENCHMARK(Sort_DPL);
#endif

static void Sort_Std(benchmark::State &state) {
  std::vector<T> a(default_vector_size);
  fill_random(a);
  Stats stats(state, sizeof(T) * default_vector_size);

  for (auto _ : state) {
    stats.rep();
    std::sort(a.begin(), a.end());
  }
}

DR_BENCHMARK(Sort_Std);
