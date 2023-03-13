// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class ForEach : public testing::Test {
public:
};

TYPED_TEST_SUITE(ForEach, TestTypes);

TYPED_TEST(ForEach, Range) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &v) { v = -v; };
  auto input = ops.vec;

  xhp::for_each(default_policy(ops.dist_vec), ops.dist_vec, negate);
  rng::for_each(ops.vec, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}
