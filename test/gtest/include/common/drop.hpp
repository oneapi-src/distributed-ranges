// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Drop : public testing::Test {
public:
};

TYPED_TEST_SUITE(Drop, AllTypes);

TYPED_TEST(Drop, Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::drop(ops.vec, 2),
                         rng::views::drop(ops.dist_vec, 2)));
}

TYPED_TEST(Drop, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(ops, rng::views::drop(ops.vec, 2),
                                rng::views::drop(ops.dist_vec, 2)));
}
