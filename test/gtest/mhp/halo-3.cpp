// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename T> class Halo3 : public testing::Test {};

TYPED_TEST_SUITE(Halo3, AllTypes);

TYPED_TEST(Halo3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mhp::default_comm().size(), 3); // dr-style ignore
}

TYPED_TEST(Halo3, dv_halos_eq) {
  TypeParam dv(10, dr::mhp::distribution().halo(2));

  iota(dv, 1);
  dv.halo().exchange();
  barrier();

  if (dr::mhp::default_comm().rank() == 0) {

    EXPECT_EQ(*(dv.begin() + 0).local(), 1);
    EXPECT_EQ(*(dv.begin() + 1).local(), 2);
    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);

  } else if (dr::mhp::default_comm().rank() == 1) {

    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
    EXPECT_EQ(*(dv.begin() + 6).local(), 7);
    EXPECT_EQ(*(dv.begin() + 7).local(), 8);

    EXPECT_EQ(*(dv.begin() + 8).local(), 9);
    EXPECT_EQ(*(dv.begin() + 9).local(), 10);

  } else {
    assert(dr::mhp::default_comm().rank() == 2);

    EXPECT_EQ(*(dv.begin() + 6).local(), 7);
    EXPECT_EQ(*(dv.begin() + 7).local(), 8);

    EXPECT_EQ(*(dv.begin() + 8).local(), 9);
    EXPECT_EQ(*(dv.begin() + 9).local(), 10);
  }
}

TYPED_TEST(Halo3, dv_different_halos_gt_first) {
  TypeParam dv(10, dr::mhp::distribution().halo(3, 1));

  iota(dv, 1);
  dv.halo().exchange();
  barrier();

  if (dr::mhp::default_comm().rank() == 0) {

    EXPECT_EQ(*(dv.begin() + 0).local(), 1);
    EXPECT_EQ(*(dv.begin() + 1).local(), 2);
    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);

  } else if (dr::mhp::default_comm().rank() == 1) {

    EXPECT_EQ(*(dv.begin() + 1).local(), 2);
    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
    EXPECT_EQ(*(dv.begin() + 6).local(), 7);
    EXPECT_EQ(*(dv.begin() + 7).local(), 8);

    EXPECT_EQ(*(dv.begin() + 8).local(), 9);

  } else {
    assert(dr::mhp::default_comm().rank() == 2);

    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
    EXPECT_EQ(*(dv.begin() + 6).local(), 7);
    EXPECT_EQ(*(dv.begin() + 7).local(), 8);

    EXPECT_EQ(*(dv.begin() + 8).local(), 9);
    EXPECT_EQ(*(dv.begin() + 9).local(), 10);
  }
}

TYPED_TEST(Halo3, dv_different_halos_gt_sec) {
  TypeParam dv(10, dr::mhp::distribution().halo(1, 3));

  iota(dv, 1);
  dv.halo().exchange();
  barrier();

  if (dr::mhp::default_comm().rank() == 0) {

    EXPECT_EQ(*(dv.begin() + 0).local(), 1);
    EXPECT_EQ(*(dv.begin() + 1).local(), 2);
    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
    EXPECT_EQ(*(dv.begin() + 6).local(), 7);

  } else if (dr::mhp::default_comm().rank() == 1) {

    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
    EXPECT_EQ(*(dv.begin() + 6).local(), 7);
    EXPECT_EQ(*(dv.begin() + 7).local(), 8);

    EXPECT_EQ(*(dv.begin() + 8).local(), 9);
    EXPECT_EQ(*(dv.begin() + 9).local(), 10);

  } else {
    assert(dr::mhp::default_comm().rank() == 2);

    EXPECT_EQ(*(dv.begin() + 7).local(), 8);

    EXPECT_EQ(*(dv.begin() + 8).local(), 9);
    EXPECT_EQ(*(dv.begin() + 9).local(), 10);
  }
}

TYPED_TEST(Halo3, dv_halos_next_0) {
  TypeParam dv(6, dr::mhp::distribution().halo(2, 0));
  std::vector<int> v(6);

  iota(dv, 1);
  dv.halo().exchange();

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(*(dv.begin() + 0).local(), 1);
    EXPECT_EQ(*(dv.begin() + 1).local(), 2);

  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(*(dv.begin() + 0).local(), 1);
    EXPECT_EQ(*(dv.begin() + 1).local(), 2);

    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);
    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
  }
}

TYPED_TEST(Halo3, dv_halos_prev_0) {
  TypeParam dv(6, dr::mhp::distribution().halo(0, 2));
  iota(dv, 1);
  dv.halo().exchange();

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(*(dv.begin() + 0).local(), 1);
    EXPECT_EQ(*(dv.begin() + 1).local(), 2);

    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(*(dv.begin() + 2).local(), 3);
    EXPECT_EQ(*(dv.begin() + 3).local(), 4);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
  } else {
    assert(dr::mhp::default_comm().rank() == 2);

    EXPECT_EQ(*(dv.begin() + 4).local(), 5);
    EXPECT_EQ(*(dv.begin() + 5).local(), 6);
  }
}
