// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DM = dr::mhp::distributed_dense_matrix<T>;
using A = std::allocator<T>;
using DMA = dr::mhp::distributed_dense_matrix<T, A>;
using DMI = typename DM::iterator;

template <typename T> class MhpTests3 : public testing::Test {};

TEST(MhpTests, DM_Create) {
  const int rows = 11, cols = 11;
  DM a(rows, cols);
  fence();

  EXPECT_EQ(a.size(), rows * cols);
}

TEST(MhpTests, DM_CreateFill) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);
  fence();

  EXPECT_EQ(*(a.begin()), -1);
  EXPECT_EQ(*(a.begin() + 13), -1);
  EXPECT_EQ(*(a.end() - 1), -1);
}

TEST(MhpTests, DM_Rows) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(3, 1, false); // 1 row
  DM a(rows, cols, -1, hb);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local()) {
      std::vector<T> v;
      rng::iota(*r, 10 * r->idx());

      v.resize(cols);
      std::iota(v.begin(), v.end(), 10 * r->idx());
      EXPECT_TRUE(equal(v, *r));
    }
  }

  // dr::mhp::for_each(b.rows(), [](auto row) { rng::iota(row, 10); });
}
