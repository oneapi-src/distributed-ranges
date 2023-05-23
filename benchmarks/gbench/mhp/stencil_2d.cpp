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

// For debugging, use  col_static = 10 with --vector-size 100
// const std::size_t cols_static = 10;
const std::size_t cols_static = 10000;

using Row = std::array<T, cols_static>;

static auto shape() {
  std::size_t rows = default_vector_size / cols_static;
  return std::pair(rows, cols_static);
}

void print_matrix(rng::forward_range auto &&actual) {
  auto [rows, cols] = shape();
  auto m = rng::views::chunk(actual, cols);
  for (auto row : m) {
    fmt::print("    ");
    for (auto e : row) {
      fmt::print("{:8.3} ", e);
    }
    fmt::print("\n");
  }
}

struct CommonChecker {
  void check(rng::forward_range auto &&actual) {
    if (init) {
      auto fp_compare = [](auto n1, auto n2) {
        return fabs(n1 - n2) / n2 < .01;
      };

      if (rng::equal(actual, expected, fp_compare)) {
        return;
      }

      fmt::print("Mismatch\n");
      if (actual.size() <= 100) {
        fmt::print("  Expected:\n");
        print_matrix(expected);
        fmt::print("  Actual:\n");
        print_matrix(actual);
      }
      exit(1);
    } else {
      auto [rows, cols] = shape();
      expected.resize(rows * cols);
      rng::copy(actual, expected.begin());
      init = true;
    }
  }

  std::vector<T> expected;
  bool init = false;
} common;

struct Checker {
  void check(rng::forward_range auto &&actual) {
    if (!check_results || checked) {
      return;
    }

    common.check(actual);
    checked = true;
  }

#ifdef SYCL_LANGUAGE_VERSION
  void check_device(sycl::queue q, T *p) {
    if (!check_results || checked) {
      return;
    }

    auto [rows, cols] = shape();
    auto sz = rows * cols;
    std::vector<T> local(sz);
    q.copy(p, local.data(), sz).wait();
    common.check(local);
    checked = true;
  }
#endif

  void check_array(rng::forward_range auto &&actual) {
    if (!check_results || checked) {
      return;
    }

    std::vector<Row> local(actual.size());
    dr::mhp::copy(0, actual, local.begin());
    auto [rows, cols] = shape();
    common.check(rng::span(&(local[0][0]), rows * cols));
    checked = true;
  }

  bool checked = false;
};

static void stencil_global_op(auto in, auto out, auto cols, auto i, auto j) {
  out[i * cols + j] =
      (in[(i - 1) * cols + j] + in[i * cols + j - 1] + in[i * cols + j] +
       in[i * cols + j + 1] + in[(i + 1) * cols + j]) /
      4;
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

  Checker checker;
  for (auto _ : state) {
    for (std::size_t i = 0; i < stencil_steps; i++) {
      for (std::size_t i = 1; i < rows - 1; i++) {
        for (std::size_t j = 1; j < cols - 1; j++) {
          stencil_global_op(in, out, cols, i, j);
        }
      }
      std::swap(in, out);
    }

    checker.check(rng::span(in, rows * cols));
  }
}

BENCHMARK(Stencil2D_Loop_Std);

auto stencil_1darray_op = [](auto &&v) {
  auto &[in_row, out_row] = v;
  auto p = &in_row;
  for (std::size_t i = 1; i < cols_static - 1; i++) {
    out_row[i] = (p[-1][i] + p[0][i - 1] + p[0][i] + p[0][i + 1] + p[1][i]) / 4;
  }
};

static void Stencil2D_1DArray_DR(benchmark::State &state) {
  auto [rows, cols] = shape();

  if (rows == 0) {
    return;
  }
  assert(cols == cols_static);

  dr::halo_bounds hb(1);
  dr::mhp::distributed_vector<Row> a(rows, hb);
  dr::mhp::distributed_vector<Row> b(rows, hb);

  auto fill_row = [](auto &row) {
    std::fill(row.begin(), row.end(), init_val);
  };
  xhp::for_each(a, fill_row);
  xhp::for_each(b, fill_row);

  Checker checker;
  auto in = rng::subrange(a.begin() + 1, a.end() - 1);
  auto out = rng::subrange(b.begin() + 1, b.end() - 1);
  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      dr::mhp::halo(in).exchange();
      dr::mhp::for_each(dr::mhp::views::zip(in, out), stencil_1darray_op);
      std::swap(in, out);
    }
    checker.check_array(stencil_steps % 2 ? b : a);
  }
}

BENCHMARK(Stencil2D_1DArray_DR);

#ifdef SYCL_LANGUAGE_VERSION
static void Stencil2D_Basic_SYCL(benchmark::State &state) {
  auto s = shape();
  auto rows = std::get<0>(s);
  auto cols = std::get<1>(s);

  if (rows == 0) {
    return;
  }

  sycl::queue q;

  auto in = sycl::malloc_device<T>(rows * cols, q);
  auto out = sycl::malloc_device<T>(rows * cols, q);
  q.fill(in, init_val, rows * cols);
  q.fill(out, init_val, rows * cols);
  q.wait();

  Checker checker;
  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      auto op = [=](auto it) {
        stencil_global_op(in, out, cols, it[0] + 1, it[1] + 1);
      };
      q.parallel_for(sycl::range(rows - 2, cols - 2), op).wait();
      std::swap(in, out);
    }
    checker.check_device(q, in);
  }
}

BENCHMARK(Stencil2D_Basic_SYCL);
#endif
