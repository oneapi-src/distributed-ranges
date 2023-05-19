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

// Store result here to avoid compiler optimization of unused
// operations
T bit_bucket = 0;

static void Fill_DR(benchmark::State &state) {
  xhp::distributed_vector<T> vec(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::fill(vec, 0);
    }
  }
}

BENCHMARK(Fill_DR);

static void Fill_Local(benchmark::State &state) {
  std::vector<T> vec(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      rng::fill(vec, 0);
    }
  }
}

BENCHMARK(Fill_Local);

static void Copy_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  xhp::distributed_vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      xhp::copy(src, dst.begin());
    }
  }
}

BENCHMARK(Copy_DR);

static void Copy_Local(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  std::vector<T> dst(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      rng::copy(src, dst.begin());
    }
  }
}

BENCHMARK(Copy_Local);

static void Reduce_DR(benchmark::State &state) {
  xhp::distributed_vector<T> src(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      bit_bucket += xhp::reduce(src);
    }
  }
}

BENCHMARK(Reduce_DR);

static void Reduce_Local(benchmark::State &state) {
  std::vector<T> src(default_vector_size);
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      bit_bucket +=
          std::reduce(std::execution::par_unseq, src.begin(), src.end());
    }
  }
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
}

BENCHMARK(TransformIdentity_Local);

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
}

BENCHMARK(Mul_DR);

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
}

BENCHMARK(Mul_Local);

static void DotProduct_DR(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size);
  xhp::distributed_vector<T> b(default_vector_size);
  auto mul = [](auto v) {
    auto [a, b] = v;
    return a * b;
  };
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      bit_bucket +=
          xhp::reduce(xhp::views::zip(a, b) | xhp::views::transform(mul));
    }
  }
}

BENCHMARK(DotProduct_DR);

static void DotProduct_Local(benchmark::State &state) {
  std::vector<T> a(default_vector_size);
  std::vector<T> b(default_vector_size);
  auto mul = [](auto v) {
    auto [a, b] = v;
    return a * b;
  };
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      auto &&m = rng::views::zip(a, b) | rng::views::transform(mul);
      bit_bucket += std::reduce(std::execution::par_unseq, m.begin(), m.end());
    }
  }
}

BENCHMARK(DotProduct_Local);

// Does not compile
#if 0
#ifdef SYCL_LANGUAGE_VERSION
static void DotProduct_DPL(benchmark::State &state) {
  auto mul = [](auto v) {
    auto [a, b] = v;
    return a * b;
  };
  sycl::queue q;
  auto policy = oneapi::dpl::execution::make_device_policy(q);
  auto a = rng::views::counted(sycl::malloc_device<T>(default_vector_size, q), default_vector_size);
  auto b = rng::views::counted(sycl::malloc_device<T>(default_vector_size, q), default_vector_size);
  auto &&z = rng::views::zip(a, b) | rng::views::transform(mul);
  dr::__detail::direct_iterator d_first(z.begin());
  dr::__detail::direct_iterator d_last(z.end());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      auto res = oneapi::dpl::experimental::reduce_async(policy, d_first, d_last, T(0), std::plus()).get();
      benchmark::DoNotOptimize(res);
    }
  }
}

BENCHMARK(DotProduct_DPL);
#endif
#endif
