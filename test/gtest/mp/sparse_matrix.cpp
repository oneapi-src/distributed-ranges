#include "xp-tests.hpp"
auto testMatrixIter(auto& src, auto &matrix) {
     EXPECT_TRUE(src.size() == matrix.size());
     auto iterCsr = src.begin();
     auto iterMatrix = matrix.begin();
     std::map<std::pair<long, long>, double> entries;
     for (auto (index, val): iterCsr) {
        entries[index] = val;
     }
     for (auto (index, val): iterMatrix) {
        EXPECT_TRUE(val == entries[index]);
     }
}

TEST(SparseMatrix, IterRow) {
  std::size_t m = 100;
  std::size_t k = 100;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
   dr::mp::distributed_sparse_matrix<
    float, unsigned long, dr::mp::MpiBackend,
    dr::mp::csr_row_distribution<float, unsigned long, dr::mp::MpiBackend>>
    a(csr, 0);
  testMatrixIter(csr, a);
}

TEST(SparseMatrix, IterEq) {
  std::size_t m = 100;
  std::size_t k = 100;
  auto csr = dr::generate_random_csr({m, k}, 0.1f);
   dr::mp::distributed_sparse_matrix<
    float, unsigned long, dr::mp::MpiBackend,
    dr::mp::csr_eq_distribution<float, unsigned long, dr::mp::MpiBackend>>
    a(csr, 0);
  testMatrixIter(csr, a);
}
