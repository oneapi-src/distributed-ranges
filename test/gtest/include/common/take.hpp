// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Take : public testing::Test {
public:
};

TYPED_TEST_SUITE(Take, TestTypes);

TYPED_TEST(Take, Basic) {
  Op1<TypeParam> op(10);

  EXPECT_TRUE(check_mutable_view(op, rng::views::take(op.v, 6),
                                 rng::views::take(op.dv, 6)));
}
