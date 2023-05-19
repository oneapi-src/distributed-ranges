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

TYPED_TEST_SUITE(MhpTests3, AllTypes);

TYPED_TEST(MhpTests3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mhp::default_comm().size(), 3); // dr-style ignore
}

TEST(MhpTests3, DM_Index) {
  assert(dr::mhp::default_comm().size() == 3);

  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);
  fence();

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], -1);
    EXPECT_EQ(a[15], -1);
    EXPECT_EQ((a.begin()[{4, 4}]), -1);
    EXPECT_EQ(a.begin()[99], -1);
    EXPECT_EQ((a.begin()[{10, 10}]), -1);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], -1);
    EXPECT_EQ(a.begin()[15], -1);
    EXPECT_EQ((a[{4, 4}]), -1);
    EXPECT_EQ(a.begin()[99], -1);
    EXPECT_EQ((a.begin()[{10, 10}]), -1);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], -1);
    EXPECT_EQ(a.begin()[15], -1);
    EXPECT_EQ((a.begin()[{4, 4}]), -1);
    EXPECT_EQ(a[99], -1);
    EXPECT_EQ((a[{10, 10}]), -1);
  }
}

TEST(MhpTests3, DM_For) {
  assert(dr::mhp::default_comm().size() == 3);
  const int rows = 11, cols = 11;
  DM a(rows, cols, -1);

  for (auto i : a) {
    i = 5;
  }

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[15], 5);
    EXPECT_EQ((a.begin()[{4, 4}]), 5);
    EXPECT_EQ(a.begin()[99], 5);
    EXPECT_EQ((a.begin()[{10, 10}]), 5);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], 5);
    EXPECT_EQ(a.begin()[15], 5);
    EXPECT_EQ((a[{4, 4}]), 5);
    EXPECT_EQ(a.begin()[99], 5);
    EXPECT_EQ((a.begin()[{10, 10}]), 5);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], 5);
    EXPECT_EQ(a.begin()[15], 5);
    EXPECT_EQ((a.begin()[{4, 4}]), 5);
    EXPECT_EQ(a[99], 5);
    EXPECT_EQ((a[{10, 10}]), 5);
  }
}

TEST(MhpTests3, DM_Fill) {
  assert(dr::mhp::default_comm().size() == 3);
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::fill(a, 5);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[15], 5);
    EXPECT_EQ((a.begin()[{4, 4}]), 5);
    EXPECT_EQ(a.begin()[99], 5);
    EXPECT_EQ((a.begin()[{10, 10}]), 5);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], 5);
    EXPECT_EQ(a.begin()[15], 5);
    EXPECT_EQ((a[{4, 4}]), 5);
    EXPECT_EQ(a.begin()[99], 5);
    EXPECT_EQ((a.begin()[{10, 10}]), 5);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], 5);
    EXPECT_EQ(a.begin()[15], 5);
    EXPECT_EQ((a.begin()[{4, 4}]), 5);
    EXPECT_EQ(a[99], 5);
    EXPECT_EQ((a[{10, 10}]), 5);
  }
}

TEST(MhpTests3, DM_Fill_HB) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(2, 3, false);
  DM a(rows, cols, -13, hb);

  dr::mhp::fill(a, 5);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[15], 5);
    EXPECT_EQ((a.begin()[{4, 4}]), 5);
    EXPECT_EQ(a.begin()[99], 5);
    EXPECT_EQ((a.begin()[{10, 10}]), 5);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], 5);
    EXPECT_EQ(a.begin()[15], 5);
    EXPECT_EQ((a[{4, 4}]), 5);
    EXPECT_EQ(a.begin()[99], 5);
    EXPECT_EQ((a.begin()[{10, 10}]), 5);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], 5);
    EXPECT_EQ(a.begin()[15], 5);
    EXPECT_EQ((a.begin()[{4, 4}]), 5);
    EXPECT_EQ(a[99], 5);
    EXPECT_EQ((a[{10, 10}]), 5);
  }
}

TEST(MhpTests3, DM_Iota1) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::iota(a, 0);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), 120);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], 0);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), 120);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], 0);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), 120);
  }
}

TEST(MhpTests3, DM_Iota2) {
  const int rows = 11, cols = 11;
  DM a(rows, cols, -13);

  dr::mhp::iota(a.begin() + 1, a.end() - 1, 1);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], -13);
    EXPECT_EQ(a[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), -13);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], -13);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), -13);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], -13);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), -13);
  }
}

TEST(MhpTests3, DM_Iota_HB) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(3, 2, false);
  DM a(rows, cols, -13, hb);

  dr::mhp::iota(a, 0);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), 120);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], 0);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), 120);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], 0);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), 120);
  }
}

TEST(MhpTests3, DM_Copy_HB) {
  const int rows = 11, cols = 11;
  dr::halo_bounds hb(3, 2, false);
  DM a(rows, cols, -13, hb);
  DM b(rows, cols, -1, hb);

  dr::mhp::iota(b, 0);
  dr::mhp::copy(b, a.begin());

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), 120);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(a.begin()[0], 0);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a[{4, 4}]), 48);
    EXPECT_EQ(a.begin()[99], 99);
    EXPECT_EQ((a.begin()[{10, 10}]), 120);
  } else /*(dr::mhp::default_comm().rank() == 2) */ {
    EXPECT_EQ(a.begin()[0], 0);
    EXPECT_EQ(a.begin()[15], 15);
    EXPECT_EQ((a.begin()[{4, 4}]), 48);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), 120);
  }
}
