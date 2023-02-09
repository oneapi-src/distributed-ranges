// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

TEST(MhpTests, IteratorConformance) {
  DV dv1(10), dv2(10);
  V v1(10);

  // 2 distributed vectors
  EXPECT_TRUE(conformant(dv1.begin(), dv2.begin()));
  ;
  // misaligned distributed vector
  EXPECT_FALSE(conformant(dv1.begin() + 1, dv2.begin()));

  // iota conformant with anything
  // EXPECT_TRUE(conformant(dv1.begin(), rng::views::iota(1)).first);
  // auto x = rng::views::iota(1).begin();
  // decltype(x)::foo = 1;
  // EXPECT_TRUE(conformant(rng::views::iota(1).begin(), dv1.begin()).first);

  // May not be useful to support
  // distributed and local vector
  // EXPECT_FALSE(conformant(dv1.begin(), v1.begin()));
}
