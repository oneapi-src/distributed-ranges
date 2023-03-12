// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Drop : public testing::Test {
public:
};

TYPED_TEST_SUITE(Drop, TestTypes);

TYPED_TEST(Drop, Basic) {
  Op1<TypeParam> op(10);

  EXPECT_TRUE(check_mutable_view(op, rng::views::drop(op.v, 2),
                                 rng::views::drop(op.dv, 2)));
}
