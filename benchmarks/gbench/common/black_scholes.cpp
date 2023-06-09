// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = double;

T normalCDF(T value) { return 0.5 * std::erfc(-value * M_SQRT1_2); }

static void Black_Scholes(benchmark::State &state) {
  T scalar = val;
  T r = 0;
  T sig = 0;

  xhp::distributed_vector<T> s0(default_vector_size, scalar);
  xhp::distributed_vector<T> x(default_vector_size, scalar);
  xhp::distributed_vector<T> t(default_vector_size, scalar);
  xhp::distributed_vector<T> vcall(default_vector_size, scalar);
  xhp::distributed_vector<T> vput(default_vector_size, scalar);

  Stats stats(state, sizeof(T) * (s0.size() + x.size() + t.size()),
              sizeof(T) * (vcall.size() + vput.size()));

  auto black_scholes = [=](auto &&e) {
    auto &&[s0, x, t, vcall, vput] = e;
    T d1 = (std::log(s0 / x) + (r + T(0.5) * sig * sig) * t) /
           (sig * std::sqrt(t));
    T d2 = (std::log(s0 / x) + (r - T(0.5) * sig * sig) * t) /
           (sig * std::sqrt(t));
    vcall = s0 * normalCDF(d1) - std::exp(-r * t) * x * normalCDF(d2);
    vput = std::exp(-r * t) * x * normalCDF(-d2) - s0 * normalCDF(-d1);
  };

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();

      xhp::for_each(xhp::views::zip(s0, x, t, vcall, vput), black_scholes);
    }
  }
}

DR_BENCHMARK(Black_Scholes);
