// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-bench.hpp"

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#endif

using T = double;

static void Fill_DR(benchmark::State &state) {
  xhp::distributed_vector<T> vec(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::fill(vec, 0);
    }
  }
  memory_bandwidth(state,
                   default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Fill_DR);

static void Fill_Local(benchmark::State &state) {
  std::vector<T> vec(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      rng::fill(vec, 0);
    }
  }
  memory_bandwidth(state,
                   default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Fill_Local);

#ifndef BENCH_SHP
// Not defined?

static void Copy_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  xhp::distributed_vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::copy(src, dst.begin());
    }
  }
  memory_bandwidth(state,
                   2 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Copy_DR);
#endif

static void Copy_Local(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  std::vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      rng::copy(src, dst.begin());
    }
  }
  memory_bandwidth(state,
                   2 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Copy_Local);

static void Reduce_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      auto res = xhp::reduce(src);
      benchmark::DoNotOptimize(res);
    }
  }
  memory_bandwidth(state,
                   default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Reduce_DR);

static void Reduce_Local(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      auto res = std::reduce(std::execution::par_unseq, src.begin(), src.end());
      benchmark::DoNotOptimize(res);
    }
  }
  memory_bandwidth(state,
                   default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Reduce_Local);

#ifdef SYCL_LANGUAGE_VERSION
static void Reduce_DPL(benchmark::State &state) {
  sycl::queue q;
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto src = sycl::malloc_device<T>(default_vector_size, q);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      auto res = std::reduce(policy, src, src + default_vector_size);
      benchmark::DoNotOptimize(res);
    }
  }
  memory_bandwidth(state,
                   default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Reduce_DPL);
#endif

static void TransformIdentity_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  xhp::distributed_vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::transform(src, dst.begin(), std::identity());
    }
  }
  memory_bandwidth(state,
                   2 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(TransformIdentity_DR);

static void TransformIdentity_Local(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  std::vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      rng::transform(src, dst.begin(), std::identity());
    }
  }
  memory_bandwidth(state,
                   2 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(TransformIdentity_Local);

#ifndef BENCH_SHP
// segfault

static void Mul_DR(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size);
  xhp::distributed_vector<T> b(default_vector_size);
  xhp::distributed_vector<T> c(default_vector_size);
  auto mul = [](auto v) {
    auto [a, b] = v;
    return a * b;
  };
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::transform(xhp::views::zip(a, b), c.begin(), mul);
    }
  }
  memory_bandwidth(state,
                   3 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Mul_DR);
#endif

static void Mul_Local(benchmark::State &state) {
  std::vector<T> a(default_vector_size);
  std::vector<T> b(default_vector_size);
  std::vector<T> c(default_vector_size);
  auto mul = [](auto a, auto b) { return a * b; };
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(),
                     c.begin(), mul);
    }
  }
  memory_bandwidth(state,
                   3 * default_repetitions * default_vector_size * sizeof(T));
}

BENCHMARK(Mul_Local);
