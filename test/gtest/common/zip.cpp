// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE(Zip, AllTypes);

// Try 1, 2, 3 to check pair/tuple issues
TYPED_TEST(Zip, Local1) {
  Ops1<TypeParam> ops(10);

  EXPECT_EQ(rng::views::zip(ops.vec), xhp::views::zip(ops.vec));
}

TYPED_TEST(Zip, Local2) {
  Ops2<TypeParam> ops(10);

  EXPECT_EQ(rng::views::zip(ops.vec0, ops.vec1),
            xhp::views::zip(ops.vec0, ops.vec1));
}

TYPED_TEST(Zip, Local3) {
  Ops3<TypeParam> ops(10);

  EXPECT_EQ(rng::views::zip(ops.vec0, ops.vec1, ops.vec2),
            xhp::views::zip(ops.vec0, ops.vec1, ops.vec2));
}

TYPED_TEST(Zip, Local3Distance) {
  Ops3<TypeParam> ops(10);

  EXPECT_EQ(rng::distance(rng::views::zip(ops.vec0, ops.vec1, ops.vec2)),
            rng::distance(xhp::views::zip(ops.vec0, ops.vec1, ops.vec2)));
}

TYPED_TEST(Zip, Local2Mutate) {
  Ops2<TypeParam> rops(10);
  Ops2<TypeParam> xops(10);

  auto r = rng::views::zip(rops.vec0, rops.vec1);
  auto x = rng::views::zip(xops.vec0, xops.vec1);
  auto n2 = [](auto v) {
    std::get<0>(v) += 1;
    std::get<1>(v) = -std::get<1>(v);
  };
  rng::for_each(r, n2);
  rng::for_each(x, n2);
  EXPECT_EQ(rops.vec0, xops.vec0);
  EXPECT_EQ(rops.vec1, xops.vec1);
}

TYPED_TEST(Zip, Dist1) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec);
  auto dist = xhp::views::zip(ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist2) {
  Ops2<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec0, ops.vec1);
  auto dist = xhp::views::zip(ops.dist_vec0, ops.dist_vec1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist3) {
  Ops3<TypeParam> ops(10);

  EXPECT_TRUE(
      check_view(rng::views::zip(ops.vec0, ops.vec1, ops.vec2),
                 xhp::views::zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

TYPED_TEST(Zip, Dist3Distance) {
  Ops3<TypeParam> ops(10);

  EXPECT_EQ(rng::distance(rng::views::zip(ops.vec0, ops.vec1, ops.vec2)),
            rng::distance(
                xhp::views::zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

TYPED_TEST(Zip, CopyConstructor) {
  Ops2<TypeParam> ops(10);

  auto dist = xhp::views::zip(ops.dist_vec0, ops.dist_vec1);
  auto dist_copy(dist);
  EXPECT_EQ(dist, dist_copy);
}

TYPED_TEST(Zip, Iota) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(xhp::views::iota(100), ops.vec);
  auto dist = xhp::views::zip(xhp::views::iota(100), ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Iota2nd) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::zip(ops.vec, xhp::views::iota(100)),
                         xhp::views::zip(ops.dist_vec, xhp::views::iota(100))));
}

#if 0
// doesn't compile in mhp
TEST(Zip, ToTransform) {
  Ops2<xhp::distributed_vector<int>> ops(10);

  auto mul = [](auto v) { return std::get<0>(v) * std::get<1>(v); };
  auto local = rng::views::transform(rng::views::zip(ops.vec0, ops.vec1), mul);
  auto dist =
      xhp::views::transform(xhp::views::zip(ops.dist_vec0, ops.dist_vec1), mul);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}
#endif

TYPED_TEST(Zip, All) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(
      rng::views::zip(rng::views::all(ops.vec0), rng::views::all(ops.vec1)),
      xhp::views::zip(rng::views::all(ops.dist_vec0),
                      rng::views::all(ops.dist_vec1))));
}

TYPED_TEST(Zip, L_Value) {
  Ops2<TypeParam> ops(10);

  auto l_value = rng::views::all(ops.vec0);
  auto d_l_value = rng::views::all(ops.dist_vec0);
  EXPECT_TRUE(
      check_view(rng::views::zip(l_value, rng::views::all(ops.vec1)),
                 xhp::views::zip(d_l_value, rng::views::all(ops.dist_vec1))));
}

TYPED_TEST(Zip, Subrange) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(
      rng::views::zip(rng::subrange(ops.vec0.begin() + 1, ops.vec0.end() - 1),
                      rng::subrange(ops.vec1.begin() + 1, ops.vec1.end() - 1)),
      xhp::views::zip(
          rng::subrange(ops.dist_vec0.begin() + 1, ops.dist_vec0.end() - 1),
          rng::subrange(ops.dist_vec1.begin() + 1, ops.dist_vec1.end() - 1))));
}

TYPED_TEST(Zip, ForEach) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(xhp::views::zip(ops.dist_vec0, ops.dist_vec1), copy);
  rng::for_each(rng::views::zip(ops.vec0, ops.vec1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Zip, ForEachIota) {
  Ops1<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(xhp::views::zip(xhp::views::iota(100), ops.dist_vec), copy);
  rng::for_each(rng::views::zip(xhp::views::iota(100), ops.vec), copy);

  EXPECT_EQ(ops.vec, ops.dist_vec);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}
