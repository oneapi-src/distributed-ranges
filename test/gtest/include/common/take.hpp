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
