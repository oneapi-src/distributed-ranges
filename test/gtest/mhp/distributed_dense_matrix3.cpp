// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename T> class MhpTests3 : public testing::Test {};

TYPED_TEST_SUITE(MhpTests3, AllTypesDM);

TYPED_TEST(MhpTests3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mhp::default_comm().size(), 3); // dr-style ignore
}

TYPED_TEST(MhpTests3, DM_Index) {
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols, -1);
  barrier();

  // local nad non-local data access
  EXPECT_EQ(a.begin()[0], -1);
  EXPECT_EQ(a.begin()[15], -1);
  EXPECT_EQ((a.begin()[{4, 4}]), -1);
  EXPECT_EQ(a.begin()[99], -1);
  EXPECT_EQ((a.begin()[{10, 10}]), -1);

  // local-only data access
  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], -1);
    EXPECT_EQ(a[15], -1);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), -1);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], -1);
    EXPECT_EQ((a[{10, 10}]), -1);
  }
}

TYPED_TEST(MhpTests3, DM_For) {
  assert(dr::mhp::default_comm().size() == 3);
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols, -1);

  for (auto i : a) {
    i = 5;
  }

  EXPECT_EQ(a.begin()[0], 5);
  EXPECT_EQ(a.begin()[15], 5);
  EXPECT_EQ((a.begin()[{4, 4}]), 5);
  EXPECT_EQ(a.begin()[99], 5);
  EXPECT_EQ((a.begin()[{10, 10}]), 5);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[15], 5);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 5);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 5);
    EXPECT_EQ((a[{10, 10}]), 5);
  }
}

TYPED_TEST(MhpTests3, DM_Fill) {
  assert(dr::mhp::default_comm().size() == 3);
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols, -13);

  dr::mhp::fill(a, 5);

  EXPECT_EQ(a.begin()[0], 5);
  EXPECT_EQ(a.begin()[15], 5);
  EXPECT_EQ((a.begin()[{4, 4}]), 5);
  EXPECT_EQ(a.begin()[99], 5);
  EXPECT_EQ((a.begin()[{10, 10}]), 5);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[15], 5);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 5);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 5);
    EXPECT_EQ((a[{10, 10}]), 5);
  }
}

TYPED_TEST(MhpTests3, DM_Fill_HB) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(2, 3);
  TypeParam a(rows, cols, -13, dist);

  dr::mhp::fill(a, 5);

  EXPECT_EQ(a.begin()[0], 5);
  EXPECT_EQ(a.begin()[15], 5);
  EXPECT_EQ((a.begin()[{4, 4}]), 5);
  EXPECT_EQ(a.begin()[99], 5);
  EXPECT_EQ((a.begin()[{10, 10}]), 5);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(a[15], 5);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 5);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 5);
    EXPECT_EQ((a[{10, 10}]), 5);
  }
}

TYPED_TEST(MhpTests3, DM_Iota_2_args) {
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols, -13);

  dr::mhp::iota(a, 0);

  EXPECT_EQ(a.begin()[0], 0);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ(a.begin()[99], 99);
  EXPECT_EQ((a.begin()[{10, 10}]), 120);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[15], 15);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 48);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), 120);
  }
}

TYPED_TEST(MhpTests3, DM_Iota_3_args) {
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols, -13);

  dr::mhp::iota(a.begin() + 1, a.end() - 1, 1);

  EXPECT_EQ(a.begin()[0], -13);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ(a.begin()[99], 99);
  EXPECT_EQ((a.begin()[{10, 10}]), -13);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], -13);
    EXPECT_EQ(a[15], 15);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 48);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), -13);
  }
}

TYPED_TEST(MhpTests3, DM_Iota_HB) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 2);
  TypeParam a(rows, cols, -13, dist);

  dr::mhp::iota(a, 0);

  barrier();

  EXPECT_EQ(a.begin()[0], 0);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ(a.begin()[99], 99);
  EXPECT_EQ((a.begin()[{10, 10}]), 120);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[15], 15);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 48);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), 120);
  }
}

TYPED_TEST(MhpTests3, DM_Copy_HB) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 2);
  TypeParam a(rows, cols, -13, dist);
  TypeParam b(rows, cols, -1, dist);

  dr::mhp::iota(b, 0);
  dr::mhp::copy(b, a.begin());

  barrier();

  EXPECT_EQ(a.begin()[0], 0);
  EXPECT_EQ(a.begin()[15], 15);
  EXPECT_EQ((a.begin()[{4, 4}]), 48);
  EXPECT_EQ(a.begin()[99], 99);
  EXPECT_EQ((a.begin()[{10, 10}]), 120);

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[15], 15);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ((a[{4, 4}]), 48);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(a[99], 99);
    EXPECT_EQ((a[{10, 10}]), 120);
  }
}
