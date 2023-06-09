// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = double;

static void Stream_Copy(benchmark::State &state) {
  T init = 0;
  xhp::distributed_vector<T> a(default_vector_size, init);
  xhp::distributed_vector<T> b(default_vector_size, init);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::for_each(xhp::views::zip(a, b),
                    [](auto &&v) { std::get<1>(v) = std::get<0>(v); });
    }
  }
}

DR_BENCHMARK(Stream_Copy);

T val = 0;

static void Stream_Scale(benchmark::State &state) {
  T scalar = val;
  xhp::distributed_vector<T> a(default_vector_size, scalar);
  xhp::distributed_vector<T> b(default_vector_size, scalar);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::for_each(xhp::views::zip(a, b), [scalar](auto &&v) {
        std::get<1>(v) = scalar * std::get<0>(v);
      });
    }
  }
}

DR_BENCHMARK(Stream_Scale);

static void Stream_Add(benchmark::State &state) {
  T scalar = val;
  xhp::distributed_vector<T> a(default_vector_size, scalar);
  xhp::distributed_vector<T> b(default_vector_size, scalar);
  xhp::distributed_vector<T> c(default_vector_size, scalar);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), sizeof(T) * c.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::for_each(xhp::views::zip(a, b, c), [](auto &&v) {
        std::get<2>(v) = std::get<0>(v) + std::get<1>(v);
      });
    }
  }
}

DR_BENCHMARK(Stream_Add);

static void Stream_Triad(benchmark::State &state) {
  T scalar = val;
  xhp::distributed_vector<T> a(default_vector_size, scalar);
  xhp::distributed_vector<T> b(default_vector_size, scalar);
  xhp::distributed_vector<T> c(default_vector_size, scalar);
  Stats stats(state, sizeof(T) * (a.size() + b.size()), sizeof(T) * c.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      xhp::for_each(xhp::views::zip(a, b, c), [scalar](auto &&v) {
        std::get<2>(v) = std::get<0>(v) + scalar * std::get<1>(v);
      });
    }
  }
}

DR_BENCHMARK(Stream_Triad);

T normalCDF(T value)
{
   return 0.5 * std::erfc(-value * M_SQRT1_2);
}

static void Black_Scholes(benchmark::State &state) {
  T scalar = val;
  T r = 0;
  T sig = 0;

  xhp::distributed_vector<T> s0(default_vector_size, scalar);
  xhp::distributed_vector<T> x(default_vector_size, scalar);
  xhp::distributed_vector<T> t(default_vector_size, scalar);
  xhp::distributed_vector<T> vcall(default_vector_size, scalar);
  xhp::distributed_vector<T> vput(default_vector_size, scalar);

  Stats stats(state, sizeof(T) * (s0.size() + x.size()), sizeof(T) * (vcall.size() + vput.size()));

  auto black_scholes =
    [=](auto&& e) {
      auto&& [s0, x, t, vcall, vput] = e;
      T d1 = ( std::log(s0/x) + (r + T(0.5)*sig*sig)*t ) /
                ( sig*std::sqrt(t) );
      T d2 = ( std::log(s0/x) + (r - T(0.5)*sig*sig)*t ) /
                ( sig*std::sqrt(t) );
      vcall = s0*normalCDF(d1) - std::exp(-r*t)*x*normalCDF(d2);
      vput = std::exp(-r*t)*x*normalCDF(-d2) - s0*normalCDF(-d1);
  };

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      
      xhp::for_each(xhp::views::zip(s0, x, t, vcall, vput), black_scholes);

    }
  }
}

DR_BENCHMARK(Black_Scholes);
