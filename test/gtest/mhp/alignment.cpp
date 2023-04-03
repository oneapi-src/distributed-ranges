// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using DV = mhp::distributed_vector<int>;

TEST(Alignment, One) {
  Ops1<DV> ops(10);
  EXPECT_TRUE(mhp::aligned(ops.dist_vec));
}

TEST(Alignment, Two) {
  Ops2<DV> ops(10);
  EXPECT_TRUE(mhp::aligned(ops.dist_vec0, ops.dist_vec1));
}

TEST(Alignment, Three) {
  Ops3<DV> ops(10);
  EXPECT_TRUE(mhp::aligned(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2));
}

TEST(Alignment, Misaligned) {
  Ops2<DV> ops(10);
  EXPECT_FALSE(mhp::aligned(rng::views::drop(ops.dist_vec0, 1), ops.dist_vec1));
}

#if 0
// Support not implemented
TEST(Alignment, Iota) {
  EXPECT_TRUE(mhp::aligned(rng::views::iota(100, 20)));
}

TEST(Alignment, Iota2) {
  Ops1<DV> ops(10);
  EXPECT_TRUE(mhp::aligned(ops.dist_vec, mhp::iota(100, 10)));
}
#endif
