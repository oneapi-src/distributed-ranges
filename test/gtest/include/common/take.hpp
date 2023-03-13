// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Take : public testing::Test {
public:
};

TYPED_TEST_SUITE(Take, TestTypes);

TYPED_TEST(Take, DISABLED_Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::take(ops.vec, 6),
                         rng::views::take(ops.dist_vec, 6)));
}

TYPED_TEST(Take, DISABLED_Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutable_view(ops, rng::views::take(ops.vec, 6),
                                 rng::views::take(ops.dist_vec, 6)));
}
