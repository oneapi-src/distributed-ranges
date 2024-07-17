// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#include <dr/mp/views/segmented.hpp>

template <typename T> class Segmented : public testing::Test {
public:
};

TYPED_TEST_SUITE(Segmented, AllTypesWithoutIshmem);

TYPED_TEST(Segmented, StaticAssert) {
  Ops1<TypeParam> ops(10);
  auto segmented = dr::mp::segmented_view(rng::views::iota(100),
                                           dr::ranges::segments(ops.dist_vec));
  static_assert(std::forward_iterator<decltype(segmented.begin())>);
  static_assert(rng::forward_range<decltype(segmented)>);
}

TYPED_TEST(Segmented, Basic) {
  Ops1<TypeParam> ops(10);
  auto segmented = dr::mp::segmented_view(rng::views::iota(100),
                                           dr::ranges::segments(ops.dist_vec));
  EXPECT_EQ(dr::ranges::segments(ops.dist_vec), segmented);
}

template <typename T> class SegmentUtils : public testing::Test {
public:
};

// traversing on host over local_segment does not work in case of both:
// device_memory and IshmemBackend (which uses device memory)
TYPED_TEST_SUITE(SegmentUtils, AllTypesWithoutIshmem);

TYPED_TEST(SegmentUtils, LocalSegment) {
  if (options.count("device-memory")) {
    return;
  }
  Ops1<TypeParam> ops(10);
  auto segments = dr::mp::local_segments(ops.dist_vec);
  auto ls = dr::mp::local_segment(ops.dist_vec);
  if (ls.size() == 0) // comparison would not be possible
    return;
  EXPECT_EQ(ls, *rng::begin(segments));
}

TYPED_TEST(SegmentUtils, OnlyRank0Data) {
  // Only first rank gets data
  TypeParam dist(10, dr::mp::distribution().granularity(10));
  EXPECT_EQ(dr::mp::local_segment(dist).empty(), comm_rank != 0);
}
