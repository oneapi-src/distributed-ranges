// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Counted : public testing::Test {
public:
};

TYPED_TEST_SUITE(Counted, AllTypes);

TYPED_TEST(Counted, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::counted(ops.vec.begin() + 1, 2);
  auto dist = xhp::views::counted(ops.dist_vec.begin() + 1, 2);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Counted, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(
      check_mutate_view(ops, rng::views::counted(ops.vec.begin() + 2, 3),
                        xhp::views::counted(ops.dist_vec.begin() + 2, 3)));
}
