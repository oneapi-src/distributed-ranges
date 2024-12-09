// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "dr/detail/coo_matrix.hpp"
#include "xp-tests.hpp"
TEST(SparseMatrix, Gemv) {
  long m = 100;
  long k = 100;

  dr::sp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}));

  dr::sp::distributed_vector<float> b(k, 1.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}

TEST(SparseMatrix, EmptyGemv) {
  std::size_t m = 100;
  std::size_t k = 100;
  using T = float;
  using I = int;

  dr::__detail::coo_matrix<T, I> base;
  auto csr = dr::__detail::convert_to_csr(base, {m, k}, base.size(),
                                          std::allocator<T>{});
  dr::sp::sparse_matrix<T, I> a =
      dr::sp::create_distributed(csr, dr::sp::row_cyclic());

  dr::sp::distributed_vector<T> b(k, 1.f);
  dr::sp::distributed_vector<T> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<T> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<T> c_ref(m, 0.f);

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}

TEST(SparseMatrix, ZeroVector) {
  std::size_t m = 100;
  std::size_t k = 100;
  using T = float;
  using I = int;
  std::vector<std::pair<std::pair<I, I>, T>> base;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      base.push_back({{i, j}, static_cast<float>(i + j)});
    }
  }

  auto csr = dr::__detail::convert_to_csr(base, {m, k}, base.size(),
                                          std::allocator<T>{});
  dr::sp::sparse_matrix<T, I> a =
      dr::sp::create_distributed(csr, dr::sp::row_cyclic());

  dr::sp::distributed_vector<T> b(k, 0.f);
  dr::sp::distributed_vector<T> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<T> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<T> c_ref(m, 0.f);

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}

TEST(SparseMatrix, NotSquareMatrix) {
  long m = 10;
  long k = 1000;

  dr::sp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}));

  dr::sp::distributed_vector<float> b(k, 1.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}

TEST(SparseMatrix, NotSquareMatrixOtherAxis) {
  long m = 1000;
  long k = 10;

  dr::sp::sparse_matrix<float> a(
      {m, k}, 0.1f,
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}));

  dr::sp::distributed_vector<float> b(k, 1.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}

TEST(SparseMatrix, VerySparseMatrix) {
  long m = 100;
  long k = 100;

  dr::sp::sparse_matrix<float> a(
      {m, k}, 0.001f,
      dr::sp::block_cyclic({dr::sp::tile::div, dr::sp::tile::div},
                           {dr::sp::nprocs(), 1}));

  dr::sp::distributed_vector<float> b(k, 1.f);
  dr::sp::distributed_vector<float> c(m, 0.f);

  dr::sp::gemv(c, a, b);

  std::vector<float> c_local(m);

  dr::sp::copy(c.begin(), c.end(), c_local.begin());

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c_local))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c_local);
}
