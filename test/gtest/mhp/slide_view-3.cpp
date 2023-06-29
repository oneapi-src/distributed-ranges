// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Slide3 : public testing::Test {};

TYPED_TEST_SUITE(Slide3, AllTypes);

TYPED_TEST(Slide3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mhp::default_comm().size(), 3);
}

// more Slide3 tests are comming, all assume that there are 3 mpi processes
TYPED_TEST(Slide3, no_sides) {
  TypeParam dv(6);
  iota(dv, 1);

  auto dv_sliding_view = xhp::views::sliding(dv, 1);
  EXPECT_EQ(rng::size(dv_sliding_view), 6);
  EXPECT_TRUE(equal({1}, dv_sliding_view[0]));
  EXPECT_TRUE(equal({2}, dv_sliding_view[1]));
  EXPECT_TRUE(equal({6}, dv_sliding_view[5]));

  const auto dv_segments = dr::ranges::segments(dv_sliding_view);

  EXPECT_EQ(3, rng::size(dv_segments));
  const auto dv_segment_0 = *dv_segments.begin();
  const auto dv_segment_1 = *(++dv_segments.begin());
  const auto dv_segment_2 = *(++(++dv_segments.begin()));

  EXPECT_EQ(2, rng::size(dv_segment_0));
  EXPECT_TRUE(equal({1}, dv_segment_0[0]));
  EXPECT_TRUE(equal({2}, dv_segment_0[1]));

  EXPECT_EQ(2, rng::size(dv_segment_1));
  EXPECT_TRUE(equal({3}, dv_segment_1[0]));
  EXPECT_TRUE(equal({4}, dv_segment_1[1]));

  EXPECT_EQ(2, rng::size(dv_segment_2));
  EXPECT_TRUE(equal({5}, dv_segment_2[0]));
  EXPECT_TRUE(equal({6}, dv_segment_2[1]));
}

TYPED_TEST(Slide3, with_sides) {
  TypeParam dv(6, dr::mhp::distribution().halo(1));
  iota(dv, 1);

  auto dv_sliding_view = xhp::views::sliding(dv, 3);
  EXPECT_EQ(rng::size(dv_sliding_view), 4);
  EXPECT_TRUE(equal({1, 2, 3}, dv_sliding_view[0]));
  EXPECT_TRUE(equal({2, 3, 4}, dv_sliding_view[1]));
  EXPECT_TRUE(equal({3, 4, 5}, dv_sliding_view[2]));
  EXPECT_TRUE(equal({4, 5, 6}, dv_sliding_view[3]));

  const auto dv_segments = dr::ranges::segments(dv_sliding_view);

  EXPECT_EQ(3, rng::size(dv_segments));
  const auto dv_segment_0 = *dv_segments.begin();
  const auto dv_segment_1 = *(++dv_segments.begin());
  const auto dv_segment_2 = *(++(++dv_segments.begin()));

  static_assert(std::same_as<decltype(dv_segment_0[0][0]),
                             dr::mhp::dv_segment_reference<TypeParam>>);

  EXPECT_EQ(1, rng::size(dv_segment_0));
  EXPECT_TRUE(equal({1, 2, 3}, dv_segment_0[0]));

  EXPECT_EQ(2, rng::size(dv_segment_1));
  EXPECT_TRUE(equal({2, 3, 4}, dv_segment_1[0]));
  EXPECT_TRUE(equal({3, 4, 5}, dv_segment_1[1]));

  EXPECT_EQ(1, rng::size(dv_segment_2));
  EXPECT_TRUE(equal({4, 5, 6}, dv_segment_2[0]));
}

TYPED_TEST(Slide3, local_no_sides_converts_to_correct_pointers) {
  TypeParam dv(6);
  iota(dv, 1);
  fence();

  auto dv_sliding_view = xhp::views::sliding(dv, 1);
  for (auto &&ls : dr::mhp::local_segments(dv_sliding_view)) {
    EXPECT_EQ(2, rng::size(ls));
    static_assert(
        std::same_as<decltype(ls[0][0]), typename TypeParam::value_type &>);
    switch (dr::mhp::default_comm().rank()) {
    case 0:
      EXPECT_TRUE(equal({1}, ls[0]));
      EXPECT_TRUE(equal({2}, ls[1]));
      break;
    case 1:
      EXPECT_TRUE(equal({3}, ls[0]));
      EXPECT_TRUE(equal({4}, ls[1]));
      break;
    case 2:
      EXPECT_TRUE(equal({5}, ls[0]));
      EXPECT_TRUE(equal({6}, ls[1]));
      break;
    default:
      EXPECT_TRUE(false);
    }
  }
}

TYPED_TEST(Slide3,
           local_converts_to_correct_pointers_with_sides_halo_eq_segment) {
  TypeParam dv(6, dr::mhp::distribution().halo(2));
  iota(dv, 1);
  dv.halo().exchange();

  auto dv_sliding_view = xhp::views::sliding(dv, 5);
  for (auto &&ls : dr::mhp::local_segments(dv_sliding_view)) {
    static_assert(
        std::same_as<decltype(ls[0][0]), typename TypeParam::value_type &>);
    switch (dr::mhp::default_comm().rank()) {
    case 0:
      EXPECT_EQ(0, rng::size(ls));
      break;
    case 1:
      EXPECT_EQ(2, rng::size(ls));
      EXPECT_TRUE(equal({1, 2, 3, 4, 5}, ls[0]));
      EXPECT_TRUE(equal({2, 3, 4, 5, 6}, ls[1]));
      break;
    case 2:
      EXPECT_EQ(0, rng::size(ls));
      break;
    default:
      EXPECT_TRUE(false);
    }
  }
}

TYPED_TEST(
    Slide3,
    local_converts_to_correct_pointers_with_sides_halo_less_than_segment) {
  TypeParam dv(6, dr::mhp::distribution().halo(1));
  iota(dv, 1);
  dv.halo().exchange();

  auto dv_sliding_view = xhp::views::sliding(dv, 3);
  for (auto &&ls : dr::mhp::local_segments(dv_sliding_view)) {
    switch (dr::mhp::default_comm().rank()) {
    case 0:
      EXPECT_EQ(1, rng::size(ls));
      EXPECT_TRUE(equal({1, 2, 3}, ls[0]));
      break;
    case 1:
      EXPECT_EQ(2, rng::size(ls));
      EXPECT_TRUE(equal({2, 3, 4}, ls[0]));
      EXPECT_TRUE(equal({3, 4, 5}, ls[1]));
      break;
    case 2:
      EXPECT_EQ(1, rng::size(ls));
      EXPECT_TRUE(equal({4, 5, 6}, ls[0]));
      break;
    default:
      EXPECT_TRUE(false);
    }
  }
}
