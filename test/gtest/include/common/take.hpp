// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
// Fixture
template <typename T> class Take : public testing::Test {
public:
};

TYPED_TEST_SUITE(Take, AllTypes);

TYPED_TEST(Take, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::take(ops.vec, 6);
  auto dist = xhp::views::take(ops.dist_vec, 6);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, rng::views::take(ops.vec, 6),
                                xhp::views::take(ops.dist_vec, 6)));
}

TYPED_TEST(Take, lessThanSize) {
  Ops1<TypeParam> ops(10);
  auto local = rng::views::take(ops.vec, 6);
  auto dist = xhp::views::take(ops.dist_vec, 6);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, sameSize) {
  Ops1<TypeParam> ops(10);
  auto local = rng::views::take(ops.vec, 10);
  auto dist = xhp::views::take(ops.dist_vec, 10);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, moreSize) {
  Ops1<TypeParam> ops(10);
  auto local = rng::views::take(ops.vec, 10);
  auto dist = xhp::views::take(ops.dist_vec, 12);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, zero) {
  Ops1<TypeParam> ops(10);
  auto local = rng::views::take(ops.vec, 0);
  auto dist = xhp::views::take(ops.dist_vec, 0);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, one) {
  Ops1<TypeParam> ops(10);
  auto local = rng::views::take(ops.vec, 1);
  auto dist = xhp::views::take(ops.dist_vec, 1);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, emptyInput_zeroSize) {
  xhp::distributed_vector<int> dv = {};
  auto dist = xhp::views::take(dv, 0);
  EXPECT_EQ(rng::size(dist), 0);
}

TYPED_TEST(Take, emptyInput_nonZeroSize) {
  xhp::distributed_vector<int> dv = {};
  auto dist = xhp::views::take(dv, 1);
  EXPECT_EQ(rng::size(dist), 0);
}

TYPED_TEST(Take, large) {
  Ops1<TypeParam> ops(1000);

  auto local = rng::views::take(ops.vec, 1000);
  auto dist = xhp::views::take(ops.dist_vec, 1000);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, takeOfOneElementHasOneSegmentAndSameRank) {
  Ops1<TypeParam> ops(10);
  auto dist_rng_view = rng::views::take(ops.dist_vec, 1);
  auto dist_xhp_view = shp::views::take(ops.dist_vec, 1);

  auto dist_rng_segments = lib::ranges::segments(dist_rng_view);
  auto dist_rng_rank = lib::ranges::rank(dist_rng_segments[0]);

  auto dist_xhp_segments = lib::ranges::segments(dist_xhp_view);
  auto dist_xhp_rank = lib::ranges::rank(dist_xhp_segments[0]);

  EXPECT_EQ(dist_rng_segments.size(), 1);
  EXPECT_EQ(dist_xhp_segments.size(), 1);
  EXPECT_EQ(dist_rng_rank, dist_xhp_rank);
}

TYPED_TEST(Take, takeOfFirstSegementHasOneSegmentAndSameRank) {
  Ops1<TypeParam> ops(10);
  auto dist_vec_seg = lib::ranges::segments(ops.dist_vec);
  auto first_seg_size = dist_vec_seg[0].size();

  auto dist_rng_view = rng::views::take(ops.dist_vec, first_seg_size);
  auto dist_rng_segments = lib::ranges::segments(dist_rng_view);
  auto rng_segment_rank = lib::ranges::rank(dist_rng_segments[0]);

  auto dist_xhp_view = shp::views::take(ops.dist_vec, first_seg_size);
  auto dist_xhp_segments = lib::ranges::segments(dist_xhp_view);
  auto xhp_segment_rank = lib::ranges::rank(dist_xhp_segments[0]);

  EXPECT_EQ(dist_rng_segments.size(), 1);
  EXPECT_EQ(dist_xhp_segments.size(), 1);
  EXPECT_EQ(rng_segment_rank, xhp_segment_rank);
}

TYPED_TEST(Take, takeOfAllButOneSizeHasAllSegmentsWithSameRanks) {
  Ops1<TypeParam> ops(10);

  auto dist_vec_seg = lib::ranges::segments(ops.dist_vec);

  auto dist_rng_view = rng::views::take(ops.dist_vec, 9);
  auto dist_xhp_view = shp::views::take(ops.dist_vec, 9);

  auto dist_rng_segments = lib::ranges::segments(dist_rng_view);
  auto dist_xhp_segments = lib::ranges::segments(dist_xhp_view);

  EXPECT_EQ(dist_vec_seg.size(), dist_rng_segments.size());
  EXPECT_EQ(dist_vec_seg.size(), dist_xhp_segments.size());

  auto segments_size = dist_rng_segments.size();

  for (auto i = 0; i <= segments_size; i++) {
    auto rng_segment_rank = lib::ranges::rank(dist_rng_segments[i]);
    auto xhp_segment_rank = lib::ranges::rank(dist_xhp_segments[i]);
    EXPECT_EQ(rng_segment_rank, xhp_segment_rank);
  }
}

TYPED_TEST(Take, takeOfMoreSizeHasSameNumberOfSegmentsAndSameRanks) {
  Ops1<TypeParam> ops(10);

  auto dist_vec_seg = lib::ranges::segments(ops.dist_vec);

  auto dist_rng_view = rng::views::take(ops.dist_vec, 11);
  auto dist_xhp_view = shp::views::take(ops.dist_vec, 11);

  auto dist_rng_segments = lib::ranges::segments(dist_rng_view);
  auto dist_xhp_segments = lib::ranges::segments(dist_xhp_view);

  EXPECT_EQ(dist_vec_seg.size(), dist_rng_segments.size());
  EXPECT_EQ(dist_vec_seg.size(), dist_xhp_segments.size());

  auto segments_size = dist_rng_segments.size();

  for (auto i = 0; i <= segments_size; i++) {
    auto rng_segment_rank = lib::ranges::rank(dist_rng_segments[i]);
    auto xhp_segment_rank = lib::ranges::rank(dist_xhp_segments[i]);
    EXPECT_EQ(rng_segment_rank, xhp_segment_rank);
  }
}
