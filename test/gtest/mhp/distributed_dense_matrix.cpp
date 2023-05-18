// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DM = dr::mhp::distributed_dense_matrix<T>;
using A = std::allocator<T>;
using DMA = dr::mhp::distributed_dense_matrix<T, A>;
using DMI = typename DM::iterator;

TEST(MhpTests, DM_Create) {
  const int rows = 11, cols = 11;
  DM a(rows, cols);

  EXPECT_EQ(a.size(), rows * cols);
}

TEST(MhpTests, DM_CreateFill) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  EXPECT_EQ(*(a.begin()), -1);
  EXPECT_EQ(*(a.begin() + 13), -1);
  EXPECT_EQ(*(a.end() - 1), -1);
}

TEST(MhpTests, DM_Index) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], -1);
  }
  EXPECT_EQ((a.begin()[{4, 4}]), -1);
  EXPECT_EQ(a.begin()[99], -1);
  EXPECT_EQ((a.begin()[{10, 10}]), -1);
}

TEST(MhpTests, DM_For) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  for (auto i : a) {
    i = 5;
  }

  EXPECT_EQ(a.begin()[0], 5);
  EXPECT_EQ(a.begin()[55], 5);
  EXPECT_EQ(a.begin()[rows * cols - 1], 5);
}

TEST(MhpTests, DM_Fill_index_1d) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::fill(a, 5);

  EXPECT_EQ(a.begin()[0], 5);
  EXPECT_EQ((a.begin()[55]), 5);
  EXPECT_EQ((a.begin()[99]), 5);
  EXPECT_EQ((a.begin()[rows * cols - 1]), 5);

}

TEST(MhpTests, DM_Fill_index_2d) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::fill(a, 5);

  EXPECT_EQ((a.begin()[{0, 0}]), 5);
  EXPECT_EQ((a.begin()[{10, 10}]), 5);
}

TEST(MhpTests, DM_Fill_HB) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(1);
  DM a(rows, cols, -13, hb);

  dr::mhp::fill(a, 5);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ((a[{0, 0}]), 5);
  }
  EXPECT_EQ((a.begin()[{10, 10}]), 5);
  
}

TEST(MhpTests, DM_Iota1) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::iota(a, 0);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
  }
  EXPECT_EQ(a.begin()[0], 0);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ((a.begin()[{10, 10}]), 120);
}

TEST(MhpTests, DM_Iota2) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::iota(a.begin(), a.end(), 0);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
  }
  EXPECT_EQ(a.begin()[0], 0);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ((a.begin()[{10, 10}]), 120);
}

TEST(MhpTests, DM_Iota_HB) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(3, 2, false);
  DM a(rows, cols, -13, hb);

  dr::mhp::iota(a, 0);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 1);
  }
  EXPECT_EQ(a.begin()[0], 0);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ((a.begin()[{10, 10}]), 120);
}
