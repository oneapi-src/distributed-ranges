// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#include <dr/mhp/views/segmented.hpp>

template <typename T> class Segmented : public testing::Test {
public:
};

TYPED_TEST_SUITE(Segmented, AllTypes);

TYPED_TEST(Segmented, StaticAssert) {
  Ops1<TypeParam> ops(10);
  auto segmented = dr::mhp::segmented_view(rng::views::iota(100),
                                           dr::ranges::segments(ops.dist_vec));
  static_assert(std::forward_iterator<decltype(segmented.begin())>);
  static_assert(rng::forward_range<decltype(segmented)>);
}

TYPED_TEST(Segmented, Basic) {
  Ops1<TypeParam> ops(10);
  auto segmented = dr::mhp::segmented_view(rng::views::iota(100),
                                           dr::ranges::segments(ops.dist_vec));
  EXPECT_EQ(dr::ranges::segments(ops.dist_vec), segmented);
}

template <typename T> class SegmentUtils : public testing::Test {
public:
};

TYPED_TEST_SUITE(SegmentUtils, AllTypes);

TYPED_TEST(SegmentUtils, LocalSegment) {
  Ops1<TypeParam> ops(10);
  auto segments = dr::mhp::local_segments(ops.dist_vec);
  EXPECT_EQ(dr::mhp::local_segment(ops.dist_vec), *rng::begin(segments));
}

TYPED_TEST(SegmentUtils, OnlyRank0Data) {
  // Only first rank gets data
  TypeParam dist(10, dr::mhp::distribution().granularity(10));
  EXPECT_EQ(dr::mhp::local_segment(dist).empty(), comm_rank != 0);
}
