// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture

template <typename T> class CopyMHP : public testing::Test {
public:
};

TYPED_TEST_SUITE(CopyMHP, AllTypes);

const std::size_t root = 0;

TYPED_TEST(CopyMHP, Dist2Local) {
  Ops2<TypeParam> ops(10);

  dr::mhp::copy(root, ops.dist_vec0, ops.vec1.begin());

  if (comm_rank == root) {
    EXPECT_EQ(ops.vec0, ops.vec1);
  }
}

TYPED_TEST(CopyMHP, Local2Dist) {
  Ops2<TypeParam> ops(10);

  dr::mhp::copy(root, ops.vec0, ops.dist_vec1.begin());

  if (comm_rank == root) {
    EXPECT_EQ(ops.vec0, ops.dist_vec1);
  }
}
