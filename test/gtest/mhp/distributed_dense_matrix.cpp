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
  barrier();

  EXPECT_EQ(a.size(), rows * cols);
}

TEST(MhpTests, DM_CreateFill) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);
  barrier();

  EXPECT_EQ(*(a.begin()), -1);
  EXPECT_EQ(*(a.begin() + 13), -1);
  EXPECT_EQ(*(a.end() - 1), -1);
}

TEST(MhpTests, DM_Rows_For) {
  const int rows = 11, cols = 11;
  dr::mhp::halo_bounds hb(3, 1, false); // 1 row
  DM a(rows, cols, -1, hb);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local()) {
      rng::iota(*r, 10 * r->idx());
    }
  }

  barrier();

  EXPECT_EQ(*(a.begin()), 0);
  EXPECT_EQ(*(a.begin() + 13), 12);
  EXPECT_EQ(*(a.begin() + 48), 44);
  EXPECT_EQ(*(a.end() - 1), 110);
  EXPECT_EQ(*(a.end() - 24), 89);
}

TEST(MhpTests, DM_Rows_ForEach) {
  const int rows = 11, cols = 11;
  dr::mhp::halo_bounds hb(3, 1, false);
  DM a(rows, cols, -1, hb);

  dr::mhp::for_each(a.rows(), [](auto row) { rng::iota(row, 10); });

  barrier();

  EXPECT_EQ(*(a.begin()), 10);
  EXPECT_EQ(*(a.begin() + 13), 12);
  EXPECT_EQ(*(a.begin() + 48), 14);
  EXPECT_EQ(*(a.end() - 1), 20);
  EXPECT_EQ(*(a.end() - 24), 19);
}

// Missing
// * transform
// * halo exchange (e.g. one step of stencil and then check exact values after
// exchange) test that dense_matrix can be instantiated and has basic operations
// * when type is not a simple int but sth more complex (like e.g. std::array)

TEST(MhpTests, DM_Transform) {
  const int rows = 11, cols = 11;
  dr::mhp::halo_bounds hb(3, 1, false);
  DM a(rows, cols, -1, hb), b(rows, cols, -1, hb);

  auto negate = [](auto v) { return -v; };

  dr::mhp::transform(a.begin(), a.end(), b.begin(), negate);

  EXPECT_EQ(*(b.begin()), 1);
  EXPECT_EQ(*(b.begin() + 13), 1);
  EXPECT_EQ(*(b.end() - 24), 1);
}

TEST(MhpTests, DM_with_std_array) {
  const int rows = 11, cols = 11;
  dr::mhp::halo_bounds hb(2);

  std::array<int, 5> ref = std::array<int, 5>({1, 2, 3, 4, 5});

  dr::mhp::distributed_dense_matrix<std::array<int, 5>> a(rows, cols, ref, hb);

  std::array<int, 5> val = *(a.begin() + 13);

  EXPECT_EQ(val[3], 4);
}

TEST(MhpTests3, DM_Halo_Exchange) {
  const int rows = 12, cols = 12;
  dr::mhp::halo_bounds hb(1, 2, false);
  DM a(rows, cols, 121, hb);

  a.halo().exchange();

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ((*(a.data() + a.get_halo_bounds().prev + a.segment_size())),
              121); // halo_bound.next area
  } else if (dr::mhp::default_comm().rank() ==
             dr::mhp::default_comm().size() - 1) {
    EXPECT_EQ((*(a.data())), 121); // halo_bound.prev area
  } else {
    EXPECT_EQ((*(a.data())), 121);
    EXPECT_EQ((*(a.data() + a.get_halo_bounds().prev + a.segment_size())), 121);
  }
}
