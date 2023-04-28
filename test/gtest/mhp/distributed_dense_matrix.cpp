// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DM = dr::mhp::distributed_dense_matrix<T>;
using A = std::allocator<T>;
using DMA = dr::mhp::distributed_dense_matrix<T, A>;
using DMI = typename DM::iterator;

TEST(MhpDmTests, DM_Create) {
  const int rows = 11, cols = 11;
  DM a(rows, cols);

  EXPECT_EQ(a.size(), rows * cols);
}

TEST(MhpDmTests, DM_CreateFill) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  EXPECT_EQ(*(a.begin()), -1);
  EXPECT_EQ(*(a.begin() + 13), -1);
  EXPECT_EQ(*(a.end() - 1), -1);
}

// TODO: index operator

/* TEST(MhpDmTests, DistributedMatrix_Index) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  EXPECT_EQ(a.begin()[55], -1);
  EXPECT_EQ(a[{4,4}], -1);

} */

/* TEST(MhpTestsDM, DM_For) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  for(auto i : a) {
    i = 5;
  }

  EXPECT_EQ(a.begin()[0], 5);
  EXPECT_EQ(a.begin()[55], 5);
  EXPECT_EQ(a.begin()[rows * cols - 1], 5);

} */

TEST(MhpTestsDM, DM_Fill) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::fill(a, 5);
  a.dump_matrix("after fill (no hb)");

  // if (comm_rank == 0)
  EXPECT_EQ(a.begin()[0], 5);
  // if (comm_rank == comm_size - 1)
  EXPECT_EQ(a.begin()[rows * cols - 1], 5);
}

TEST(MhpTestsDM, DM_Fill_HB) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(1);
  DM a(rows, cols, -13, hb);

  dr::mhp::fill(a, 5);
  a.dump_matrix("after fill");

  // if (comm_rank == 0)
  EXPECT_EQ(a.begin()[0], 5);
  // if (comm_rank == comm_size - 1)
  EXPECT_EQ(a.begin()[rows * cols - 1], 5);
}
