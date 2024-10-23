// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

// Fixture
template <typename T> class Count : public testing::Test {
protected:
};

TYPED_TEST_SUITE(Count, AllTypes);

TYPED_TEST(Count, EmptyIf) {
  std::vector<int> vec;

  Ops1<TypeParam> ops(0);

  auto pred = [=](auto &&v) { return true; };

  EXPECT_EQ(xp::count_if(ops.dist_vec, pred), 0);
  EXPECT_EQ(std::count_if(ops.vec.begin(), ops.vec.end(), pred),
            xp::count_if(ops.dist_vec, pred));
}

TYPED_TEST(Count, BasicFirstElem) {
  std::vector<int> vec{1, 2, 3, 1, 1, 3, 4, 1, 5, 6, 7};

  Ops1<TypeParam> ops(vec.size());
  ops.vec = vec;
  xp::copy(ops.vec, ops.dist_vec.begin());

  auto value = *ops.vec.begin();

  EXPECT_EQ(xp::count(ops.dist_vec, value), 4);
  EXPECT_EQ(std::count(ops.vec.begin(), ops.vec.end(), value),
            xp::count(ops.dist_vec, value));
}

TYPED_TEST(Count, BasicFirstElemIf) {
  std::vector<int> vec{1, 2, 3, 1, 1, 3, 4, 1, 5, 6, 7};

  Ops1<TypeParam> ops(vec.size());
  ops.vec = vec;
  xp::copy(ops.vec, ops.dist_vec.begin());

  auto value = *vec.begin();
  auto pred = [=](auto &&v) { return v == value; };

  EXPECT_EQ(xp::count_if(ops.dist_vec, pred), 4);
  EXPECT_EQ(std::count_if(ops.vec.begin(), ops.vec.end(), pred),
            xp::count_if(ops.dist_vec, pred));
}

TYPED_TEST(Count, FirstElemsIf) {
  std::vector<int> vec(20);
  std::iota(vec.begin(), vec.end(), 0);

  Ops1<TypeParam> ops(vec.size());
  ops.vec = vec;
  xp::copy(ops.vec, ops.dist_vec.begin());

  auto pred = [=](auto &&v) { return v < 5; };

  EXPECT_EQ(xp::count_if(ops.dist_vec, pred), 5);
  EXPECT_EQ(std::count_if(ops.vec.begin(), ops.vec.end(), pred),
            xp::count_if(ops.dist_vec, pred));
}
