// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class Equals : public testing::Test {
public:
};

TYPED_TEST_SUITE(Equals, AllTypes);

TYPED_TEST(Equals, Same) {
  Ops1<TypeParam> ops(10);

  xhp::distributed_vector<int> toCompareXhp(10);
  std::vector<int> toCompareStd(10);

  for (std::size_t idx = 0; idx < 10; idx++) {
    toCompareXhp[idx] = ops.dist_vec[idx];
    toCompareStd[idx] = ops.vec[idx];
  }
  barrier();

  bool xhpEq = xhp::equal(ops.dist_vec, toCompareXhp);
  bool stdEq = rng::equal(ops.vec, toCompareStd);

  EXPECT_TRUE(xhpEq == stdEq);
}

TYPED_TEST(Equals, Different) {
  Ops1<TypeParam> ops(10);

  xhp::distributed_vector<int> toCompareXhp(10);
  std::vector<int> toCompareStd(10);

  for (std::size_t idx = 0; idx < 10; idx++) {
    toCompareXhp[idx] = ops.dist_vec[idx];
    toCompareStd[idx] = ops.vec[idx];
  }

  toCompareXhp[2] = -ops.dist_vec[2];
  toCompareStd[2] = -ops.vec[2];

  barrier();

  bool xhpEq = xhp::equal(ops.dist_vec, toCompareXhp);
  bool stdEq = rng::equal(ops.vec, toCompareStd);

  EXPECT_TRUE(xhpEq == stdEq);
}
