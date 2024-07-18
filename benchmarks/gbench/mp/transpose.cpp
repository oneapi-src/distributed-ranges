// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "oneapi/mkl/blas.hpp"

#include "../common/dr_bench.hpp"

using T = double;

class TransposeFixture : public benchmark::Fixture {
protected:
  std::array<std::size_t, 2> shape2d;
  std::size_t rows, cols, size2d, bytes2d;

public:
  void SetUp(::benchmark::State &) {
    std::size_t n = ::sqrtl(default_vector_size);
    rows = n;
    cols = n;
    shape2d = {rows, cols};
    size2d = shape2d[0] * shape2d[1];
    bytes2d = size2d * sizeof(T);
  }

  void TearDown(::benchmark::State &) {}
};

using Mdarray = dr::mp::distributed_mdarray<T, 2>;

void show2d(const std::string &title, Mdarray &mat) {
  fmt::print("{}:\n", title);
  for (std::size_t i = 0; i < 4; i++) {
    for (std::size_t j = 0; j < 4; j++) {
      fmt::print("{}  ", mat.mdspan()(i, j));
    }
    fmt::print("\n");
  }
}

BENCHMARK_DEFINE_F(TransposeFixture, Transpose2D_DR)(benchmark::State &state) {
  Mdarray in(shape2d), out(shape2d);
  xhp::iota(in, 100);
  xhp::iota(out, 200);

  Stats stats(state, bytes2d, bytes2d);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      dr::mp::transpose(in, out);
    }
  }
  // show2d("dr out", out);
}

DR_BENCHMARK_REGISTER_F(TransposeFixture, Transpose2D_DR);

namespace row_major = oneapi::mkl::blas::row_major;

void mkl_transpose(sycl::queue q, std::size_t rows, std::size_t cols, T *A,
                   std::size_t lda, T *B, std::size_t ldb) {
  oneapi::mkl::blas::row_major::omatcopy(q, oneapi::mkl::transpose::trans, rows,
                                         cols, T(0), A, lda, B, ldb)
      .wait();
}

BENCHMARK_DEFINE_F(TransposeFixture, Transpose2D_Reference)
(benchmark::State &state) {
  sycl::queue q;
  auto in = sycl::malloc_device<T>(size2d, q);
  auto out = sycl::malloc_device<T>(size2d, q);
  q.fill(in, T(1), size2d);
  q.fill(out, T(2), size2d);
  q.wait();

  Stats stats(state, bytes2d, bytes2d);

  // mkl omatcopy
  // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/omatcopy.html
  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      mkl_transpose(q, rows, cols, in, cols, out, rows);
    }
  }

  sycl::free(in, q);
  sycl::free(out, q);
}

DR_BENCHMARK_REGISTER_F(TransposeFixture, Transpose2D_Reference);
