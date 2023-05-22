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

static const T init_val = 1;

const std::size_t cols_static = 10000;

static auto shape() {
  std::size_t rows = default_vector_size / cols_static;
  return std::pair(rows, cols_static);
}

static void Stencil2D_Loop_Std(benchmark::State &state) {
  auto [rows, cols] = shape();
  if (rows == 0) {
    return;
  }
  std::vector<T> a(rows * cols, init_val);
  std::vector<T> b(rows * cols, init_val);

  auto in = a.data();
  auto out = b.data();

  for (auto _ : state) {
    for (std::size_t i = 0; i < stencil_steps; i++) {
      for (std::size_t i = 1; i < rows - 1; i++) {
        for (std::size_t j = 1; j < cols - 1; j++) {
          out[i * cols + j] =
              (in[(i - 1) * cols] + in[i * cols + j - 1] + in[i * cols + j] +
               in[i * cols + j + 1] + in[(i + 1) * cols + j]) /
              5;
        }
      }
      std::swap(in, out);
    }
  }
}

BENCHMARK(Stencil2D_Loop_Std);

auto stencil_1darray_op = [](auto &&v) {
  auto &[in_row, out_row] = v;
  auto p = &in_row;
  for (std::size_t i = 1; i < cols_static - 1; i++) {
    out_row[i] = p[-1][i] + p[0][i - 1] + p[0][i] + p[0][i + 1] + p[1][i];
  }
};

static void Stencil2D_1DArray_DR(benchmark::State &state) {
  auto [rows, cols] = shape();

  if (rows == 0) {
    return;
  }
  assert(cols == cols_static);

  using Row = std::array<T, cols_static>;

  dr::halo_bounds hb(1);
  dr::mhp::distributed_vector<Row> a(rows, hb);
  dr::mhp::distributed_vector<Row> b(rows, hb);

  auto fill_row = [](auto &row) {
    std::fill(row.begin(), row.end(), init_val);
  };
  xhp::for_each(a, fill_row);
  xhp::for_each(b, fill_row);

  auto in = rng::subrange(a.begin() + 1, a.end() - 1);
  auto out = rng::subrange(b.begin() + 1, b.end() - 1);
  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      dr::mhp::halo(in).exchange();
      dr::mhp::for_each(dr::mhp::views::zip(in, out), stencil_1darray_op);
      std::swap(in, out);
    }
  }
}

BENCHMARK(Stencil2D_1DArray_DR);
