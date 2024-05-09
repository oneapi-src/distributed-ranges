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

static const T init_val = 1;

// For debugging, use  col_static = 10 with --vector-size 100
// const std::size_t cols_static = 10;
const std::size_t cols_static = 10000;

using Row = std::array<T, cols_static>;

static auto default_shape() {
  std::size_t rows = default_vector_size / cols_static;
  return std::array{rows, cols_static};
}

static auto shape(auto &state) {
  auto s = default_shape();
  if (s[0] < 3) {
    state.SkipWithError(fmt::format("Vector size must be >= 3 * {}", s[1]));
    std::size_t empty = 0;
    return std::array{empty, empty};
  }
  return s;
}

void print_matrix(rng::forward_range auto &&actual) {
  auto s = default_shape();
  auto cols = s[1];
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

      if (!rng::equal(actual, expected, fp_compare)) {
        fmt::print("Mismatch\n");
        fmt::print("Actual size: {}\n", actual.size());
        if (actual.size() <= 100) {
          fmt::print("  Expected:\n");
          print_matrix(expected);
          fmt::print("  Actual:\n");
          print_matrix(actual);
        }
        exit(1);
      }
    } else {
      auto s = default_shape();
      auto rows = s[0];
      auto cols = s[1];
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

    auto s = default_shape();
    auto rows = s[0];
    auto cols = s[1];
    auto sz = rows * cols;
    std::vector<T> local(sz);
    q.copy(p, local.data(), sz).wait();
    common.check(local);
    checked = true;
  }
#endif // SYCL_LANGUAGE_VERSION

  void check_array(rng::forward_range auto &&actual) {
    if (!check_results || checked) {
      return;
    }

    std::vector<Row> local(actual.size());
    dr::mhp::copy(0, actual, local.begin());
    auto s = default_shape();
    auto rows = s[0];
    auto cols = s[1];
    common.check(rng::span(&(local[0][0]), rows * cols));
    checked = true;
  }

  bool checked = false;
};

static void stencil_1darray_op(auto in, auto out, auto cols, auto i, auto j) {
  out[i * cols + j] =
      (in[(i - 1) * cols + j] + in[i * cols + j - 1] + in[i * cols + j] +
       in[i * cols + j + 1] + in[(i + 1) * cols + j]) /
      4;
}

//
// Serial baseline
//
static void Stencil2D_Loop_Serial(benchmark::State &state) {
  auto s = default_shape();
  auto rows = s[0];
  auto cols = s[1];
  if (rows == 0) {
    return;
  }
  std::vector<T> a(rows * cols, init_val);
  std::vector<T> b(rows * cols, init_val);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  auto in = a.data();
  auto out = b.data();

  Checker checker;
  for (auto _ : state) {
    for (std::size_t i = 0; i < stencil_steps; i++) {
      stats.rep();
      for (std::size_t i = 1; i < rows - 1; i++) {
        for (std::size_t j = 1; j < cols - 1; j++) {
          stencil_1darray_op(in, out, cols, i, j);
        }
      }
      std::swap(in, out);
    }

    checker.check(rng::span(in, rows * cols));
  }
}

DR_BENCHMARK(Stencil2D_Loop_Serial);

auto stencil_foreach_stdArray_op = [](auto &&v) {
  auto &[in_row, out_row] = v;
  auto p = &in_row;
  for (std::size_t i = 1; i < cols_static - 1; i++) {
    out_row[i] = (p[-1][i] + p[0][i - 1] + p[0][i] + p[0][i + 1] + p[1][i]) / 4;
  }
};

//
// Distributed vector of std::array
//
static void Stencil2D_ForeachStdArray_DR(benchmark::State &state) {
  auto s = default_shape();
  auto rows = s[0];
  auto cols = s[1];

  if (rows == 0) {
    return;
  }
  assert(cols == cols_static);

  using Row = std::array<T, cols_static>;

  auto dist = dr::mhp::distribution().halo(1);
  dr::mhp::distributed_vector<Row> a(rows, dist);
  dr::mhp::distributed_vector<Row> b(rows, dist);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

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
      stats.rep();
      dr::mhp::halo(in).exchange();
      dr::mhp::for_each(dr::mhp::views::zip(in, out),
                        stencil_foreach_stdArray_op);
      std::swap(in, out);
    }
    checker.check_array(stencil_steps % 2 ? b : a);
  }
}

DR_BENCHMARK(Stencil2D_ForeachStdArray_DR);

//
// Distributed vector of floats. Granularity ensures segments contain
// whole rows. Explicitly process segments SPMD-style.
//
static void Stencil2D_Segmented_DR(benchmark::State &state) {
  auto s = default_shape();
  auto rows = s[0];
  auto cols = s[1];

  if (rows == 0) {
    return;
  }

  auto dist = dr::mhp::distribution().halo(cols).granularity(cols);
  dr::mhp::distributed_vector<T> a(rows * cols, init_val, dist);
  dr::mhp::distributed_vector<T> b(rows * cols, init_val, dist);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  Checker checker;
  auto in =
      dr::mhp::local_segment(rng::subrange(a.begin() + cols, a.end() - cols));
  auto out =
      dr::mhp::local_segment(rng::subrange(b.begin() + cols, b.end() - cols));
  auto size = rng::size(in);
  assert(size % cols == 0);
  auto row_slice = size / cols;
  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      stats.rep();
      dr::mhp::halo(stencil_steps % 2 ? b : a).exchange();
      for (std::size_t i = 0; i < row_slice; i++) {
        for (std::size_t j = 1; j < cols - 1; j++) {
          stencil_1darray_op(in.begin(), out.begin(), cols, i, j);
        }
      }
      std::swap(in, out);
    }
    checker.check(stencil_steps % 2 ? b : a);
  }
}

DR_BENCHMARK(Stencil2D_Segmented_DR);

#if __GNUC__ == 10 && __GNUC_MINOR__ == 4
// mdspan triggers gcc 10 bugs, skip these tests
#else

//
// Distributed vector of floats. Granularity ensures segments contain
// whole rows. Explicitly process segments SPMD-style.
//
static void Stencil2D_Tiled_DR(benchmark::State &state) {
  auto s = default_shape();

  if (s[0] == 0) {
    return;
  }

  auto dist = dr::mhp::distribution().halo(1);
  dr::mhp::distributed_mdarray<T, 2> a(s, dist);
  dr::mhp::distributed_mdarray<T, 2> b(s, dist);
  xhp::fill(a, init_val);
  xhp::fill(b, init_val);

  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  Checker checker;
  auto in = a.grid()(comm_rank, 0).mdspan();
  auto out = b.grid()(comm_rank, 0).mdspan();

  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      stats.rep();
      dr::mhp::halo(stencil_steps % 2 ? b : a).exchange();
      std::size_t first = 0 + (comm_rank == 0);
      std::size_t last = in.extent(0) - (comm_rank == (ranks - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = 1; j < in.extent(1) - 1; j++) {
          out(i, j) = (in(i - 1, j) + in(i, j - 1) + in(i, j) + in(i, j + 1) +
                       in(i + 1, j)) /
                      4;
        }
      }
      std::swap(in, out);
    }
    checker.check(stencil_steps % 2 ? b : a);
  }
}

DR_BENCHMARK(Stencil2D_Tiled_DR);

auto mdspan_stencil_op = [](auto v) {
  auto [in, out] = v;
#if 0
  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      fmt::print("{:4} ", in(i, j));
    }
    fmt::print("\n");
  }
  fmt::print("\n");
#endif
  out(0, 0) = (in(-1, 0) + in(0, -1) + in(0, 0) + in(0, 1) + in(1, 0)) / 4;
};

//
// Distributed vector of floats. Granularity ensures segments contain
// whole rows. Explicitly process segments SPMD-style.
//
static void Stencil2D_MdspanTuple_DR(benchmark::State &state) {
  auto shape = default_shape();

  if (shape[0] == 0) {
    return;
  }
  auto dist = dr::mhp::distribution().halo(1);
  dr::mhp::distributed_mdarray<T, 2> a(shape, dist);
  dr::mhp::distributed_mdarray<T, 2> b(shape, dist);
  xhp::fill(a, init_val);
  xhp::fill(b, init_val);

  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  Checker checker;
  auto in = a.grid()(comm_rank, 0).mdspan();
  auto out = b.grid()(comm_rank, 0).mdspan();

  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      stats.rep();
      dr::mhp::halo(stencil_steps % 2 ? b : a).exchange();
      std::size_t first = 0 + (comm_rank == 0);
      std::size_t last = in.extent(0) - (comm_rank == (ranks - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = 1; j < in.extent(1) - 1; j++) {
          auto in_stencil = md::mdspan(std::to_address(&in(i, j)), shape);
          auto out_stencil = md::mdspan(std::to_address(&out(i, j)), shape);
          mdspan_stencil_op(std::tuple(in_stencil, out_stencil));
        }
      }
      std::swap(in, out);
    }
    checker.check(stencil_steps % 2 ? b : a);
  }
}

DR_BENCHMARK(Stencil2D_MdspanTuple_DR);

//
// Distributed vector of floats. Granularity ensures segments contain
// whole rows. Explicitly process segments SPMD-style.
//
static void Stencil2D_DR(benchmark::State &state) {
  auto shape = default_shape();
  std::size_t radius = 1;
  std::array slice_starts{radius, radius};
  std::array slice_ends{shape[0] - radius, shape[1] - radius};
  if (shape[0] == 0) {
    return;
  }

  auto dist = dr::mhp::distribution().halo(radius);
  dr::mhp::distributed_mdarray<T, 2> a(shape, dist);
  dr::mhp::distributed_mdarray<T, 2> b(shape, dist);
  xhp::fill(a, init_val);
  xhp::fill(b, init_val);

  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  auto in = dr::mhp::views::submdspan(a.view(), slice_starts, slice_ends);
  auto out = dr::mhp::views::submdspan(b.view(), slice_starts, slice_ends);
  auto in_array = &a;
  auto out_array = &b;

  Checker checker;
  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      stats.rep();
      dr::mhp::halo(*in_array).exchange();
      xhp::stencil_for_each(mdspan_stencil_op, in, out);
      std::swap(in, out);
      std::swap(in_array, out_array);
    }
    checker.check(*in_array);
  }
}

DR_BENCHMARK(Stencil2D_DR);

#endif //__GNUC__ == 10 && __GNUC_MINOR__ == 4

auto round_up(auto n, auto multiple) {
  return (n + multiple - 1) / multiple * multiple;
}

//
// Single process SYCL baseline
//
#ifdef SYCL_LANGUAGE_VERSION
static void Stencil2D_Reference(benchmark::State &state) {
  auto s = default_shape();
  auto rows = s[0];
  auto cols = s[1];

  if (rows == 0) {
    return;
  }
  sycl::range local(16, 16);
  sycl::nd_range nd_range(
      sycl::range(round_up(rows - 2, local[0]), round_up(cols - 2, local[1])),
      local);

  sycl::queue q = dr::mhp::sycl_queue();

  auto in = sycl::malloc_device<T>(rows * cols, q);
  auto out = sycl::malloc_device<T>(rows * cols, q);
  Stats stats(state, sizeof(T) * rows * cols, sizeof(T) * rows * cols);
  q.fill(in, init_val, rows * cols);
  q.fill(out, init_val, rows * cols);
  q.wait();

  Checker checker;
  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      stats.rep();
      auto op = [=](auto nd_idx) {
        std::size_t i = nd_idx.get_global_id(0) + 1;
        std::size_t j = nd_idx.get_global_id(1) + 1;
        if (i < rows - 1 && j < cols - 1) {
          stencil_1darray_op(in, out, cols, i, j);
        }
      };
      q.parallel_for(nd_range, op).wait();
      std::swap(in, out);
    }
    checker.check_device(q, in);
  }
  sycl::free(in, q);
  sycl::free(out, q);
}

DR_BENCHMARK(Stencil2D_Reference);

//
// Distributed vector of floats. Granularity ensures segments contain
// whole rows. Explicitly process segments SPMD-style with SYCL
//
[[maybe_unused]] static void
Stencil2D_SegmentedSYCL_DR(benchmark::State &state) {
  auto s = default_shape();
  auto rows = s[0];
  auto cols = s[1];

  if (rows == 0) {
    return;
  }

  auto dist = dr::mhp::distribution().halo(cols).granularity(cols);
  dr::mhp::distributed_vector<T> a(rows * cols, init_val, dist);
  dr::mhp::distributed_vector<T> b(rows * cols, init_val, dist);
  Stats stats(state, sizeof(T) * a.size(), sizeof(T) * b.size());

  // fails on devcloud
  // Checker checker;
  auto in =
      dr::mhp::local_segment(rng::subrange(a.begin() + cols, a.end() - cols));
  auto out =
      dr::mhp::local_segment(rng::subrange(b.begin() + cols, b.end() - cols));
  auto size = rng::size(in);
  assert(size % cols == 0);
  auto row_slice = size / cols;

  auto q = dr::mhp::sycl_queue();
  sycl::range global(row_slice, cols - 2);

  for (auto _ : state) {
    for (std::size_t s = 0; s < stencil_steps; s++) {
      stats.rep();
      auto op = [=](auto it) {
        stencil_1darray_op(in, out, cols, it[0], it[1] + 1);
      };
      dr::mhp::halo(stencil_steps % 2 ? b : a).exchange();
      q.parallel_for(sycl::range(row_slice, cols - 2), op).wait();
      std::swap(in, out);
    }
    // fails on devcloud
    // checker.check(stencil_steps % 2 ? b : a);
  }
}
// disabled due to DRA-136
// DR_BENCHMARK(Stencil2D_SegmentedSYCL_DR);

#endif // SYCL_LANGUAGE_VERSION
