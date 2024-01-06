// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../common/dr_bench.hpp"

using T = double;

static void MdspanUtil_Pack(benchmark::State &state) {
  std::vector<T> a(num_rows * num_columns);
  std::vector<T> b(num_rows * num_columns);

  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      dr::__detail::mdspan_copy(
          md::mdspan(a.data(), std::array{num_rows, num_columns}), b.begin())
          .wait();
    }
  }
}

DR_BENCHMARK(MdspanUtil_Pack);
