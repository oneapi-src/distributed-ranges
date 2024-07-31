// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

// Fixture
template <typename T> class Count : public testing::Test {
protected:
};

TYPED_TEST_SUITE(Count, AllTypes);

TYPED_TEST(Count, BasicFirstElem) {
  Ops1<TypeParam> ops(10);
  auto value = *ops.vec.begin();

  EXPECT_EQ(std::count(ops.vec.begin(), ops.vec.end(), value),
            xp::count(ops.dist_vec, value));
}

TYPED_TEST(Count, BasicFirstElemIf) {
  Ops1<TypeParam> ops(10);
  auto value = *ops.vec.begin();
  auto pred = [=](auto &&v) { return v == value; };

  EXPECT_EQ(std::count_if(ops.vec.begin(), ops.vec.end(), pred),
            xp::count_if(ops.dist_vec, pred));
}

TYPED_TEST(Count, FirstElemsIf) {
  Ops1<TypeParam> ops(10);
  auto pred = [=](auto &&v) { return v < 5; };

  EXPECT_EQ(std::count_if(ops.vec.begin(), ops.vec.end(), pred),
            xp::count_if(ops.dist_vec, pred));
}
