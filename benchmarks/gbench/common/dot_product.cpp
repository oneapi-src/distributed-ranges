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

static T init_val = 1;

void check_dp(auto actual, const nostd::source_location location =
                               nostd::source_location::current()) {
  auto expected = default_vector_size * init_val * init_val;
  if (expected != actual) {
    fmt::print("Error in {}\n"
               "  Expected: {}\n"
               "  Actual: {}\n",
               location.function_name(), expected, actual);
    exit(1);
  } else {
    return;
  }
}

static void DotProduct_DR(benchmark::State &state) {
  xhp::distributed_vector<T> a(default_vector_size, init_val);
  xhp::distributed_vector<T> b(default_vector_size, init_val);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), 0);
  auto mul = [](auto v) {
    auto [a, b] = v;
    return a * b;
  };
  T res = 0;
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      res = xhp::reduce(xhp::views::zip(a, b) | xhp::views::transform(mul));
      benchmark::DoNotOptimize(res);
    }
  }
  check_dp(res);
}

DR_BENCHMARK(DotProduct_DR);

#ifdef SYCL_LANGUAGE_VERSION
static void DotProduct_Reference(benchmark::State &state) {
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);

  auto ap = sycl::malloc_device<T>(default_vector_size, q);
  auto bp = sycl::malloc_device<T>(default_vector_size, q);
  std::span<T> a(ap, default_vector_size);
  std::span<T> b(bp, default_vector_size);

  q.fill(ap, init_val, a.size());
  q.fill(bp, init_val, b.size());
  q.wait();

  Stats stats(state, sizeof(T) * (a.size() + b.size()), 0);
  auto z = rng::views::zip(a, b) | rng::views::transform([](auto &&elem) {
             return std::get<0>(elem) * std::get<1>(elem);
           });
  dr::__detail::direct_iterator d_first(z.begin());
  dr::__detail::direct_iterator d_last(z.end());

  T res = 0;
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      res = std::reduce(policy, d_first, d_last, T(0), std::plus());
      benchmark::DoNotOptimize(res);
    }
  }
  check_dp(res);
  sycl::free(ap, q);
  sycl::free(bp, q);
}

DR_BENCHMARK(DotProduct_Reference);
#endif
