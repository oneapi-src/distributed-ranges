// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class All : public testing::Test {
public:
};

TYPED_TEST_SUITE(All, AllTypes);

TYPED_TEST(All, Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(
      check_view(rng::views::all(ops.vec), rng::views::all(ops.dist_vec)));
}

TYPED_TEST(All, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, rng::views::all(ops.vec),
                                rng::views::all(ops.dist_vec)));
}
