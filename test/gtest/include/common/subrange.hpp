// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Subrange : public testing::Test {
public:
};

TYPED_TEST_SUITE(Subrange, TestTypes);

TYPED_TEST(Subrange, Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(
      rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1),
      rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1)));
}

TYPED_TEST(Subrange, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutable_view(
      ops, rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1),
      rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1)));
}
