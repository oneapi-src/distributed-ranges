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

using T = float;
T initval = 0;

T normalCDF(T value) { return 0.5 * std::erfc(-value * M_SQRT1_2); }

auto black_scholes = [](auto r, auto sig, auto &&e) {
  auto &&[s0, x, t, vcall, vput] = e;
  T d1 =
      (std::log(s0 / x) + (r + T(0.5) * sig * sig) * t) / (sig * std::sqrt(t));
  T d2 =
      (std::log(s0 / x) + (r - T(0.5) * sig * sig) * t) / (sig * std::sqrt(t));
  vcall = s0 * normalCDF(d1) - std::exp(-r * t) * x * normalCDF(d2);
  vput = std::exp(-r * t) * x * normalCDF(-d2) - s0 * normalCDF(-d1);
};

static void BlackScholes_DR(benchmark::State &state) {
  T scalar = initval;
  T r = 0;
  T sig = 0;

  xhp::distributed_vector<T> s0(default_vector_size, scalar);
  xhp::distributed_vector<T> x(default_vector_size, scalar);
  xhp::distributed_vector<T> t(default_vector_size, scalar);
  xhp::distributed_vector<T> vcall(default_vector_size, scalar);
  xhp::distributed_vector<T> vput(default_vector_size, scalar);

  Stats stats(state, sizeof(T) * (s0.size() + x.size() + t.size()),
              sizeof(T) * (vcall.size() + vput.size()));

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();

      xhp::for_each(xhp::views::zip(s0, x, t, vcall, vput),
                    [r, sig, bs = black_scholes](auto e) { bs(r, sig, e); });
    }
  }
}

DR_BENCHMARK(BlackScholes_DR);

#ifdef SYCL_LANGUAGE_VERSION
static void BlackScholes_Reference(benchmark::State &state) {
  T r = 0;
  T sig = 0;
  auto q = get_queue();
  auto policy = oneapi::dpl::execution::make_device_policy(q);

  auto s0 = sycl::malloc_device<T>(default_vector_size, q);
  auto x = sycl::malloc_device<T>(default_vector_size, q);
  auto t = sycl::malloc_device<T>(default_vector_size, q);
  auto vcall = sycl::malloc_device<T>(default_vector_size, q);
  auto vput = sycl::malloc_device<T>(default_vector_size, q);
  q.fill(s0, initval, default_vector_size);
  q.fill(x, initval, default_vector_size);
  q.fill(t, initval, default_vector_size);
  q.fill(vcall, initval, default_vector_size);
  q.fill(vput, initval, default_vector_size);
  q.wait();

  std::span<T> d_s0(s0, default_vector_size);
  std::span<T> d_x(x, default_vector_size);
  std::span<T> d_t(t, default_vector_size);
  std::span<T> d_vcall(vcall, default_vector_size);
  std::span<T> d_vput(vput, default_vector_size);

  Stats stats(state, sizeof(T) * 3 * default_vector_size,
              sizeof(T) * 2 * default_vector_size);

  auto zipped = rng::views::zip(d_s0, d_x, d_t, d_vcall, d_vput);
  dr::__detail::direct_iterator d_first(zipped.begin());
  dr::__detail::direct_iterator d_last(zipped.end());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      std::for_each(policy, d_first, d_last,
                    [r, sig, bs = black_scholes](auto e) { bs(r, sig, e); });
    }
  }

  sycl::free(s0, q);
  sycl::free(x, q);
  sycl::free(t, q);
  sycl::free(vcall, q);
  sycl::free(vput, q);
}

DR_BENCHMARK(BlackScholes_Reference);
#endif
