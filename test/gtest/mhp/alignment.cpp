// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = typename DV::iterator;

TEST(MhpTests, IteratorConformance) {
  DV dv1(10), dv2(10);
  V v1(10);

  // 2 distributed vectors
  EXPECT_TRUE(aligned(dv1.begin(), dv2.begin()));
  EXPECT_TRUE(aligned(dv1.begin(), dv2.begin(), dv1.begin()));
  ;
  // misaligned distributed vector
  EXPECT_FALSE(aligned(dv1.begin() + 1, dv2.begin()));
  EXPECT_FALSE(aligned(dv1.begin() + 1, dv2.begin(), dv2.begin()));
  EXPECT_FALSE(aligned(dv2.begin(), dv1.begin() + 1, dv2.begin()));

  auto aligned_z = mhp::views::zip(dv1, dv2);
  auto misaligned_z = mhp::views::zip(dv1, dv2 | rng::views::drop(1));
  EXPECT_TRUE(mhp::aligned(aligned_z.begin()));
  EXPECT_FALSE(mhp::aligned(misaligned_z.begin()));

  // iota aligned with anything
  // EXPECT_TRUE(aligned(dv1.begin(), rng::views::iota(1)).first);
  // auto x = rng::views::iota(1).begin();
  // decltype(x)::foo = 1;
  // EXPECT_TRUE(aligned(rng::views::iota(1).begin(), dv1.begin()).first);

  // May not be useful to support
  // distributed and local vector
  // EXPECT_FALSE(aligned(dv1.begin(), v1.begin()));
}
