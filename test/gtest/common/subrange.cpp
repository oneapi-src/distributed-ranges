// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Subrange : public testing::Test {
public:
};

TYPED_TEST_SUITE(Subrange, AllTypes);

TYPED_TEST(Subrange, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1);
  auto dist = rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Subrange, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_view(
      ops, rng::subrange(ops.vec.begin() + 1, ops.vec.end() - 1),
      rng::subrange(ops.dist_vec.begin() + 1, ops.dist_vec.end() - 1)));
}
