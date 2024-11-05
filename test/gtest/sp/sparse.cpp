// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "xp-tests.hpp"

TEST(SparseMatrix, IterationForward) {
  std::size_t m = 10;
  std::size_t k = 10;
  using T = float;
  using I = int;
  std::vector<std::pair<std::pair<I, I>, T>> base;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      base.push_back({{i, j}, static_cast<float>(i + j)});
    }
  }
  std::vector<std::pair<std::pair<I, I>, T>> reference(base.size());
  std::copy(base.begin(), base.end(), reference.begin());
  auto csr = dr::__detail::convert_to_csr(base, {m, k}, base.size(),
                                              std::allocator<T>{});
  dr::sp::sparse_matrix<T, I> a =
      dr::sp::create_distributed(csr, dr::sp::row_cyclic());
  int i = 0;
  for (auto elem : a) {
    auto [index, value] = elem;
    auto [real_index, real_value] = reference[i];
    auto [m, n] = index;
    auto [r_m, r_n] = real_index;

    EXPECT_TRUE(m == r_m && n == r_n) << fmt::format(
        "Reference m, n:\n  {}, {}\nActual:\n  {}, {}\n", r_m, r_n, m, n);
    EXPECT_TRUE(value == real_value) << fmt::format(
        "Reference value:\n  {}\nActual:\n  {}\n", real_value, value);
    i++;
  }
}

TEST(SparseMatrix, IterationReverse) {
  std::size_t m = 10;
  std::size_t k = 10;
  using T = float;
  using I = int;
  std::vector<std::pair<std::pair<I, I>, T>> base;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      base.push_back({{i, j}, static_cast<float>(i + j)});
    }
  }
  std::vector<std::pair<std::pair<I, I>, T>> reference(base.size());
  std::copy(base.begin(), base.end(), reference.begin());
  auto csr = dr::__detail::convert_to_csr(base, {m, k}, base.size(),
                                              std::allocator<T>{});
  dr::sp::sparse_matrix<T, I> a =
      dr::sp::create_distributed(csr, dr::sp::row_cyclic());
  int i = base.size();
  auto iterator = a.end();
  while (iterator > a.begin()) {
    iterator--;
    i--;
    auto [index, value] = *iterator;
    auto [real_index, real_value] = reference[i];
    auto [m, n] = index;
    auto [r_m, r_n] = real_index;
    EXPECT_TRUE(m == r_m && n == r_n) << fmt::format(
        "Reference m, n:\n  {}, {}\nActual:\n  {}, {}\n", r_m, r_n, m, n);
    EXPECT_TRUE(value == real_value) << fmt::format(
        "Reference value:\n  {}\nActual:\n  {}\n", real_value, value);
  }
}
