// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class ExclusiveScan : public testing::Test {
public:
};

TYPED_TEST_SUITE(ExclusiveScan, AllTypes);

TYPED_TEST(ExclusiveScan, whole_range) {
  TypeParam dv_in(5);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(5, 0);

  xhp::exclusive_scan(dv_in, dv_out, std::plus<>(), 10);
  EXPECT_EQ(10, dv_out[0]);
  EXPECT_EQ(10 + 1, dv_out[1]);
  EXPECT_EQ(10 + 1 + 2, dv_out[2]);
  EXPECT_EQ(10 + 1 + 2 + 3, dv_out[3]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4, dv_out[4]);
}
