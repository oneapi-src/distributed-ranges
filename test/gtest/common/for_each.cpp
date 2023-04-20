// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class ForEach : public testing::Test {
public:
};

TYPED_TEST_SUITE(ForEach, AllTypes);

TYPED_TEST(ForEach, Range) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &v) { v = -v; };
  auto input = ops.vec;

  xhp::for_each(ops.dist_vec, negate);
  rng::for_each(ops.vec, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}

TYPED_TEST(ForEach, Iterators) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto &v) { v = -v; };
  auto input = ops.vec;

  xhp::for_each(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1, negate);
  rng::for_each(ops.vec.begin() + 1, ops.vec.end() - 1, negate);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}
