// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename T> class MhpTests : public testing::Test {};

TYPED_TEST_SUITE(MhpTests, AllTypesDM);

TYPED_TEST(MhpTests, DM_Create) {
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols);
  dr::mhp::barrier();

  EXPECT_EQ(a.size(), rows * cols);
}

TYPED_TEST(MhpTests, DM_CreateFill) {
  const int rows = 11, cols = 11;
  TypeParam a(rows, cols, -1);
  dr::mhp::barrier();

  EXPECT_EQ(*(a.begin()), -1);
  EXPECT_EQ(*(a.begin() + 13), -1);
  EXPECT_EQ(*(a.end() - 1), -1);
}

TYPED_TEST(MhpTests, DM_Rows_For) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 1);
  TypeParam a(rows, cols, -1, dist);

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

TYPED_TEST(MhpTests, DM_Rows_ForEach) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 1);
  TypeParam a(rows, cols, -1, dist);

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

TYPED_TEST(MhpTests, DM_Transform) {
  const int rows = 11, cols = 11;
  auto dist = dr::mhp::distribution().halo(3, 1);
  TypeParam a(rows, cols, -1, dist), b(rows, cols, -1, dist);

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

  barrier();

  std::array<int, 5> val01 = *(a.begin() + 1);
  std::array<int, 5> val10 = *(a.begin() + 10);

  EXPECT_EQ(val01[3], 4);
  EXPECT_EQ(val10[1], 2);
}

TYPED_TEST(MhpTests, DM_Halo_Exchange) {
  const int rows = 12, cols = 12;
  auto dist = dr::mhp::distribution().halo(2, 2);
  TypeParam a(rows, cols, 121, dist);

  auto rank = dr::mhp::default_comm().rank();
  auto size = dr::mhp::default_comm().size();

  // no sense to test with 1 node - no exchange takes place

  if (size == 1)
    return;

  auto halo_prev_beg = a.data();
  auto halo_next_beg = a.data() + a.halo_bounds().prev + a.segment_size();

  dr::mhp::fill(a, -1);
  dr::mhp::barrier();

  if (rank == 0) {
    EXPECT_EQ(*halo_next_beg, 121);
    EXPECT_EQ(*(halo_next_beg + 21), 121);
  } else if (rank < size - 1) {
    EXPECT_EQ(*halo_prev_beg, 121);
    EXPECT_EQ(*(halo_prev_beg + 21), 121);
    EXPECT_EQ(*halo_next_beg, 121);
    EXPECT_EQ(*(halo_next_beg + 21), 121);
  } else {
    assert(rank == size - 1);
    EXPECT_EQ(*halo_prev_beg, 121);
    EXPECT_EQ(*(halo_prev_beg + 21), 121);
  }

  a.halo().exchange();

  if (rank == 0) {
    EXPECT_EQ(*halo_next_beg, -1);
    EXPECT_EQ(*(halo_next_beg + 21), -1);
  } else if (rank < size - 1) {
    EXPECT_EQ(*halo_prev_beg, -1);
    EXPECT_EQ(*(halo_prev_beg + 21), -1);
    EXPECT_EQ(*halo_next_beg, -1);
    EXPECT_EQ(*(halo_next_beg + 21), -1);
  } else {
    assert(rank == size - 1);
    EXPECT_EQ(*halo_prev_beg, -1);
    EXPECT_EQ(*(halo_prev_beg + 21), -1);
  }
}

TYPED_TEST(MhpTests, DM_Halo_Exchange_assymetric_1_2) {
  const int rows = 12, cols = 12;
  auto dist = dr::mhp::distribution().halo(1, 2);
  TypeParam a(rows, cols, 121, dist);

  auto rank = dr::mhp::default_comm().rank();
  auto size = dr::mhp::default_comm().size();

  // no sense to test with 1 node - no exchange takes place

  if (size == 1)
    return;

  auto halo_prev_beg = a.data();
  auto halo_next_beg = a.data() + a.halo_bounds().prev + a.segment_size();

  dr::mhp::fill(a, -1);
  dr::mhp::barrier();

  if (rank == 0) {
    EXPECT_EQ(*halo_next_beg, 121);
    EXPECT_EQ(*(halo_next_beg + 21), 121);
  } else if (rank < size - 1) {
    EXPECT_EQ(*halo_prev_beg, 121);
    EXPECT_EQ(*(halo_prev_beg + 11), 121);
    EXPECT_EQ(*halo_next_beg, 121);
    EXPECT_EQ(*(halo_next_beg + 21), 121);
  } else {
    assert(rank == size - 1);
    EXPECT_EQ(*halo_prev_beg, 121);
    EXPECT_EQ(*(halo_prev_beg + 11), 121);
  }

  a.halo().exchange();

  if (rank == 0) {
    EXPECT_EQ(*halo_next_beg, -1);
    EXPECT_EQ(*(halo_next_beg + 21), -1);
  } else if (rank < size - 1) {
    EXPECT_EQ(*halo_prev_beg, -1);
    EXPECT_EQ(*(halo_prev_beg + 11), -1);
    EXPECT_EQ(*halo_next_beg, -1);
    EXPECT_EQ(*(halo_next_beg + 21), -1);
  } else {
    assert(rank == size - 1);
    EXPECT_EQ(*halo_prev_beg, -1);
    EXPECT_EQ(*(halo_prev_beg + 11), -1);
  }
}

TYPED_TEST(MhpTests, DM_Halo_Exchange_assymetric_2_1) {
  const int rows = 12, cols = 12;
  auto dist = dr::mhp::distribution().halo(2, 1);
  TypeParam a(rows, cols, 121, dist);

  auto rank = dr::mhp::default_comm().rank();
  auto size = dr::mhp::default_comm().size();

  // no sense to test with 1 node - no exchange takes place

  if (size == 1)
    return;

  auto halo_prev_beg = a.data();
  auto halo_next_beg = a.data() + a.halo_bounds().prev + a.segment_size();

  dr::mhp::fill(a, -1);
  dr::mhp::barrier();

  if (rank == 0) {
    EXPECT_EQ(*halo_next_beg, 121);
    EXPECT_EQ(*(halo_next_beg + 11), 121);
  } else if (rank < size - 1) {
    EXPECT_EQ(*halo_prev_beg, 121);
    EXPECT_EQ(*(halo_prev_beg + 21), 121);
    EXPECT_EQ(*halo_next_beg, 121);
    EXPECT_EQ(*(halo_next_beg + 11), 121);
  } else {
    assert(rank == size - 1);
    EXPECT_EQ(*halo_prev_beg, 121);
    EXPECT_EQ(*(halo_prev_beg + 21), 121);
  }

  a.halo().exchange();

  if (rank == 0) {
    EXPECT_EQ(*halo_next_beg, -1);
    EXPECT_EQ(*(halo_next_beg + 11), -1);
  } else if (rank < size - 1) {
    EXPECT_EQ(*halo_prev_beg, -1);
    EXPECT_EQ(*(halo_prev_beg + 21), -1);
    EXPECT_EQ(*halo_next_beg, -1);
    EXPECT_EQ(*(halo_next_beg + 11), -1);
  } else {
    assert(rank == size - 1);
    EXPECT_EQ(*halo_prev_beg, -1);
    EXPECT_EQ(*(halo_prev_beg + 21), -1);
  }
}
