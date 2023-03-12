// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Reduce : public testing::Test {
protected:
};

TYPED_TEST_SUITE(Reduce, TestTypes);

auto reduce_basic(const auto &op) {
  return std::reduce(op.v.begin(), op.v.end(), 3, std::plus{});
}

auto reduce_default_op(const auto &op) {
  return std::reduce(op.v.begin(), op.v.end(), 2);
}

auto reduce_default_init(const auto &op) {
  return std::reduce(op.v.begin(), op.v.end());
}

TYPED_TEST(Reduce, Range) {
  Op1<TypeParam> op(10);

  EXPECT_EQ(reduce_basic(op),
            xhp::reduce(default_policy(op.dv), op.dv, 3, std::plus{}));
}

TYPED_TEST(Reduce, Iterators) {
  Op1<TypeParam> op(10);

  EXPECT_EQ(reduce_basic(op), xhp::reduce(default_policy(op.dv), op.dv.begin(),
                                          op.dv.end(), 3, std::plus{}));
}

TYPED_TEST(Reduce, RangeDefaultOp) {
  Op1<TypeParam> op(10);

  EXPECT_EQ(reduce_default_op(op),
            xhp::reduce(default_policy(op.dv), op.dv, 2));
}

TYPED_TEST(Reduce, IteratorsDefaultOp) {
  Op1<TypeParam> op(10);

  EXPECT_EQ(reduce_default_op(op),
            xhp::reduce(default_policy(op.dv), op.dv.begin(), op.dv.end(), 2));
}

TYPED_TEST(Reduce, RangeDefaultInit) {
  Op1<TypeParam> op(10);

  EXPECT_EQ(reduce_default_init(op), xhp::reduce(default_policy(op.dv), op.dv));
}

TYPED_TEST(Reduce, IteratorsDefaultInit) {
  Op1<TypeParam> op(10);

  EXPECT_EQ(reduce_default_init(op),
            xhp::reduce(default_policy(op.dv), op.dv.begin(), op.dv.end()));
}
