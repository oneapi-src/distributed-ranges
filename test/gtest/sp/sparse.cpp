
#include "xp-tests.hpp"

TEST(SparseMatrix, Iteration) {
  std::size_t m = 100;
  std::size_t k = 100;
  using T = float;
  using I = int;
  std::vector<std::pair<std::pair<I,I>,T>> base;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      base.push_back({{i, j}, i + j});
    }
  }

  auto csr = dr::sp::__detail::convert_to_csr(base, {m,k}, base.size(), std::allocator<T>{});
  dr::sp::sparse_matrix<T, I> a = dr::sp::create_distributed(
      csr,
      dr::sp::row_cyclic());
  int i = 0;
  for (auto elem : a) {
    auto [index, value] = elem;
    auto [real_index, real_value] = base[i];
    EXPECT_TRUE(value == real_value)
        << fmt::format("Reference:\n  {}\nActual:\n  {}\n", real_value, value);
    i++;
  }
  auto iterator = a.end();
  while (iterator > a.begin()) {
    iterator--;
    i--;
    auto [index, value] = *iterator;
    auto [real_index, real_value] = base[i];
    EXPECT_TRUE(value == real_value)
        << fmt::format("Reference:\n  {}\nActual:\n  {}\n", real_value, value);
    i--;
  }
}