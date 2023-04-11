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
  auto dist = rng::views::take(ops.dist_vec, 6);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Take, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, rng::views::take(ops.vec, 6),
                                rng::views::take(ops.dist_vec, 6)));
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

TYPED_TEST(Take, takeOfOneElementHasOneSegmentAndSameRank) {}
TYPED_TEST(Take, takeOfFirstSegementHasOneSegmentAndSameRank) {}
TYPED_TEST(Take, takeOfAllButOneSizeHasAllSegmentsWithSameRanks) {} //(use large input here, e.g. 1000 elements)
TYPED_TEST(Take, takeOfMoreSizeHasSameNumberOfSegmentsAndSameRanks) {}