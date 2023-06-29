// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DM = dr::mhp::distributed_dense_matrix<T>;

template <typename T> class MhpTests3 : public testing::Test {};

TEST(MhpTests, DM_Create) {
  const int rows = 11, cols = 11;
  DM a(rows, cols);
  dr::mhp::barrier();

  EXPECT_EQ(a.size(), rows * cols);
}

TEST(MhpTests, DM_CreateFill) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);
  dr::mhp::barrier();

  EXPECT_EQ(*(a.begin()), -1);
  EXPECT_EQ(*(a.begin() + 13), -1);
  EXPECT_EQ(*(a.end() - 1), -1);
}

TEST(MhpTests, DM_Rows_For) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 1);
  DM a(rows, cols, -1, dist);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local()) {
      rng::iota(*r, 10 * r->idx());
    }
  }

  dr::mhp::barrier();

  EXPECT_EQ(*(a.begin()), 0);
  EXPECT_EQ(*(a.begin() + 13), 12);
  EXPECT_EQ(*(a.begin() + 48), 44);
  EXPECT_EQ(*(a.end() - 1), 110);
  EXPECT_EQ(*(a.end() - 24), 89);
}

TEST(MhpTests, DM_Rows_ForEach) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 1);
  DM a(rows, cols, -1, dist);

  dr::mhp::for_each(a.rows(), [](auto row) { rng::iota(row, 10); });

  dr::mhp::barrier();

  EXPECT_EQ(*(a.begin()), 10);
  EXPECT_EQ(*(a.begin() + 13), 12);
  EXPECT_EQ(*(a.begin() + 48), 14);
  EXPECT_EQ(*(a.end() - 1), 20);
  EXPECT_EQ(*(a.end() - 24), 19);
}

// Missing
// * halo exchange (e.g. one step of stencil and then check exact values after
// exchange) test that dense_matrix can be instantiated and has basic operations

TEST(MhpTests, DM_Transform) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 1);
  DM a(rows, cols, -1, dist), b(rows, cols, -1, dist);

  auto negate = [](auto v) { return -v; };

  dr::mhp::transform(a.begin(), a.end(), b.begin(), negate);

  dr::mhp::barrier();

  EXPECT_EQ(*(b.begin()), 1);
  EXPECT_EQ(*(b.begin() + 13), 1);
  EXPECT_EQ(*(b.end() - 24), 1);
}

TEST(MhpTests, DM_with_std_array) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(2);

  std::array<int, 5> ref = std::array<int, 5>({1, 2, 3, 4, 5});

  dr::mhp::distributed_dense_matrix<std::array<int, 5>> a(rows, cols, ref,
                                                          dist);

  std::array<int, 5> val = *(a.begin() + 13);

  barrier();

  EXPECT_EQ(val[3], 4);
}
