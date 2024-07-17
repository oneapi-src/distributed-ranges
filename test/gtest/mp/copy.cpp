// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture

template <typename T> class CopyMP : public testing::Test {
public:
};

TYPED_TEST_SUITE(CopyMP, AllTypes);

const std::size_t root = 0;

TYPED_TEST(CopyMP, Dist2Local) {
  Ops2<TypeParam> ops(10);

  dr::mp::copy(root, ops.dist_vec0, ops.vec1.begin());

  if (comm_rank == root) {
    EXPECT_EQ(ops.vec0, ops.vec1);
  }
}

TYPED_TEST(CopyMP, Local2Dist) {
  Ops2<TypeParam> ops(10);

  dr::mp::copy(root, ops.vec0, ops.dist_vec1.begin());

  if (comm_rank == root) {
    EXPECT_EQ(ops.vec0, ops.dist_vec1);
  }
}
