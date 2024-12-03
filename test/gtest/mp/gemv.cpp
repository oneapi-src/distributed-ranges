// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"
auto testMatrixGemv(std::size_t m, std::size_t k, auto &a) {
  std::vector<float> base_b(k, 1.f);
  std::vector<float> c(m, 0.f);

  dr::mp::broadcasted_vector<float> allocated_b;
  allocated_b.broadcast_data(k, 0, base_b, dr::mp::default_comm());

  dr::mp::gemv(c, a, allocated_b);

  std::vector<float> c_ref(m, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    c_ref[i] += v;
  }

  EXPECT_TRUE(fp_equal(c_ref, c))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c);
}

auto testMatrixGemm(std::size_t m, std::size_t n, auto &a, std::size_t width) {
  std::vector<float> base_b(n * width);
  std::vector<float> c(m * width, 0.f);

  for (auto i = 0; i < n * width; i++) {
    base_b[i] = i;
  }

  dr::mp::broadcasted_slim_matrix<float> allocated_b;
  allocated_b.broadcast_data(n, width, 0, base_b, dr::mp::default_comm());

  dr::mp::gemv(c, a, allocated_b);

  std::vector<float> c_ref(m * width, 0.f);

  for (auto &&[index, v] : a) {
    auto &&[i, k] = index;

    for (auto j = 0; j < width; j++) {
      c_ref[i + j * m] += v * base_b[k + j * n];
    }
  }

  EXPECT_TRUE(fp_equal(c_ref, c))
      << fmt::format("Reference:\n  {}\nActual:\n  {}\n", c_ref, c);
}

TEST(SparseMatrix, GemvRow) {
  std::size_t m = 100;
  std::size_t k = 100;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemv(m, k, a);
}

TEST(SparseMatrix, GemvEq) {
  std::size_t m = 100;
  std::size_t k = 100;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemv(m, k, a);
}

TEST(SparseMatrix, GemvRowNotSquare) {
  std::size_t m = 1000;
  std::size_t k = 10;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemv(m, k, a);
}

TEST(SparseMatrix, GemvEqNotSquare) {
  std::size_t m = 1000;
  std::size_t k = 10;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemv(m, k, a);
}

TEST(SparseMatrix, GemvRowNotSquareDifferentAxis) {
  std::size_t m = 10;
  std::size_t k = 1000;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemv(m, k, a);
}

TEST(SparseMatrix, GemvEqNotSquareDifferentAxis) {
  std::size_t m = 10;
  std::size_t k = 1000;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemv(m, k, a);
}

TEST(SparseMatrix, GemmRow) {
  std::size_t m = 100;
  std::size_t k = 100;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemm(m, k, a, 20);
}

TEST(SparseMatrix, GemmEq) {
  std::size_t m = 100;
  std::size_t k = 100;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemm(m, k, a, 20);
}

TEST(SparseMatrix, GemmRowNotSquare) {
  std::size_t m = 1000;
  std::size_t k = 10;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemm(m, k, a, 20);
}

TEST(SparseMatrix, GemmEqNotSquare) {
  std::size_t m = 1000;
  std::size_t k = 10;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemm(m, k, a, 20);
}

TEST(SparseMatrix, GemmRowNotSquareDifferentAxis) {
  std::size_t m = 10;
  std::size_t k = 1000;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemm(m, k, a, 20);
}

TEST(SparseMatrix, GemmEqNotSquareDifferentAxis) {
  std::size_t m = 10;
  std::size_t k = 1000;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
  dr::mp::distributed_sparse_matrix<
      float, unsigned long, dr::mp::MpiBackend,
      dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
      a(csr, 0);
  testMatrixGemm(m, k, a, 20);
}
