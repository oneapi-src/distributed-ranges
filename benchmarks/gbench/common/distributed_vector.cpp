// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/async>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#endif

using T = double;

static void Fill_DR(benchmark::State &state) {
  T init = 0;
  xhp::distributed_vector<T> a(default_vector_size, init);
  Stats stats(state, 0, sizeof(T) * a.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::fill(a, init);
    }
  }
}

DR_BENCHMARK(Fill_DR);

static void Fill_Serial(benchmark::State &state) {
  T init = 0;
  std::vector<T> a(default_vector_size, init);
  Stats stats(state, 0, sizeof(T) * a.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      rng::fill(a, 0);
    }
  }
}

DR_BENCHMARK(Fill_Serial);

#ifdef SYCL_LANGUAGE_VERSION

static void Fill_QueueFill_SYCL(benchmark::State &state) {
  sycl::queue q;
  T init = 0;
  auto dst = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, 0, sizeof(T) * default_vector_size);
  q.fill(dst, init, default_vector_size).wait();
  for (auto _ : state) {
    stats.rep();
    q.fill(dst, init, default_vector_size).wait();
  }
}

DR_BENCHMARK(Fill_QueueFill_SYCL);

static void Fill_ParallelFor_SYCL(benchmark::State &state) {
  sycl::queue q;
  T init = 0;
  auto dst = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, 0, sizeof(T) * default_vector_size);
  q.fill(dst, init, default_vector_size).wait();
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      q.parallel_for(default_vector_size, [=](auto index) {
         dst[index] = init;
       }).wait();
    }
  }
}

DR_BENCHMARK(Fill_ParallelFor_SYCL);

static void Copy_ParallelFor_SYCL(benchmark::State &state) {
  sycl::queue q;
  T init = 0;
  auto src = sycl::malloc_device<T>(default_vector_size, q);
  auto dst = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);
  q.fill(src, init, default_vector_size).wait();
  q.fill(dst, init, default_vector_size).wait();
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      q.parallel_for(default_vector_size, [=](auto index) {
         dst[index] = src[index];
       }).wait();
    }
  }
}

DR_BENCHMARK(Copy_ParallelFor_SYCL);

static void Copy_QueueCopy_SYCL(benchmark::State &state) {
  sycl::queue q;
  T init = 0;
  auto src = sycl::malloc_device<T>(default_vector_size, q);
  auto dst = sycl::malloc_device<T>(default_vector_size, q);
  q.fill(src, init, default_vector_size).wait();
  q.fill(dst, init, default_vector_size).wait();
  Stats stats(state, sizeof(T) * default_vector_size,
              sizeof(T) * default_vector_size);
  for (auto _ : state) {
    stats.rep();
    q.copy(dst, src, default_vector_size).wait();
  }
}

DR_BENCHMARK(Copy_QueueCopy_SYCL);

#endif

static void Copy_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  xhp::distributed_vector<T> dst(default_vector_size);
  Stats stats(state, sizeof(T) * src.size(), sizeof(T) * dst.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::copy(src, dst.begin());
    }
  }
}

DR_BENCHMARK(Copy_DR);

static void Copy_Serial(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  std::vector<T> dst(default_vector_size);
  Stats stats(state, sizeof(T) * src.size(), sizeof(T) * dst.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      rng::copy(src, dst.begin());
    }
  }
}

DR_BENCHMARK(Copy_Serial);

static void Reduce_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  Stats stats(state, sizeof(T) * src.size(), 0);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      auto res = xhp::reduce(src);
      benchmark::DoNotOptimize(res);
    }
  }
}

DR_BENCHMARK(Reduce_DR);

static void Reduce_Serial(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  Stats stats(state, sizeof(T) * src.size(), 0);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      auto res = std::reduce(std::execution::par_unseq, src.begin(), src.end());
      benchmark::DoNotOptimize(res);
    }
  }
}

DR_BENCHMARK(Reduce_Serial);

#ifdef SYCL_LANGUAGE_VERSION
static void Reduce_DPL(benchmark::State &state) {
  sycl::queue q;
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto src = sycl::malloc_device<T>(default_vector_size, q);
  Stats stats(state, sizeof(T) * default_vector_size, 0);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      auto res = std::reduce(policy, src, src + default_vector_size);
      benchmark::DoNotOptimize(res);
    }
  }
}

DR_BENCHMARK(Reduce_DPL);
#endif

static void TransformIdentity_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  xhp::distributed_vector<T> dst(default_vector_size);
  Stats stats(state, sizeof(T) * src.size(), sizeof(T) * dst.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::transform(src, dst.begin(), std::identity());
    }
  }
}

DR_BENCHMARK(TransformIdentity_DR);

static void TransformIdentity_Serial(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  std::vector<T> dst(default_vector_size);
  Stats stats(state, sizeof(T) * src.size(), sizeof(T) * dst.size());
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      rng::transform(src, dst.begin(), std::identity());
    }
  }
}

DR_BENCHMARK(TransformIdentity_Serial);

#ifndef BENCH_SHP
// segfault

static void Mul_DR(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size);
  xhp::distributed_vector<T> b(default_vector_size);
  xhp::distributed_vector<T> c(default_vector_size);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), sizeof(T) * c.size());
  auto mul = [](auto v) {
    auto [a, b] = v;
    return a * b;
  };
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::transform(xhp::views::zip(a, b), c.begin(), mul);
    }
  }
}

DR_BENCHMARK(Mul_DR);
#endif

static void Mul_Serial(benchmark::State &state) {
  std::vector<T> a(default_vector_size);
  std::vector<T> b(default_vector_size);
  std::vector<T> c(default_vector_size);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), sizeof(T) * c.size());
  auto mul = [](auto a, auto b) { return a * b; };
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(),
                     c.begin(), mul);
    }
  }
}

DR_BENCHMARK(Mul_Serial);

static void Inclusive_Scan(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size, 3);
  xhp::distributed_vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::inclusive_scan(a, b);
    }
  }
}

DR_BENCHMARK(Inclusive_Scan);

static void Inclusive_Scan_Serial(benchmark::State &state) {
  std::vector<T> a(default_vector_size, 3);
  std::vector<T> b(default_vector_size, 0);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::inclusive_scan(std::execution::par_unseq, rng::begin(a), rng::end(a),
                          rng::begin(b));
    }
  }
}

DR_BENCHMARK(Inclusive_Scan_Serial);
