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
  EXPECT_TRUE(aligned(dv1, dv2));
  EXPECT_TRUE(aligned(dv1, dv2, dv1));
  ;
  // misaligned distributed vector
  auto udv1 = rng::views::drop(dv1, 1);
  EXPECT_FALSE(aligned(udv1, dv2));
  EXPECT_FALSE(aligned(udv1, dv2, dv2));
  EXPECT_FALSE(aligned(dv2, udv1, dv2));

  auto aligned_z = mhp::views::zip(dv1, dv2);
  auto misaligned_z = mhp::views::zip(dv1, dv2 | rng::views::drop(1));
  EXPECT_TRUE(mhp::aligned(aligned_z));
  EXPECT_FALSE(mhp::aligned(misaligned_z));

  // iota aligned with anything
  // EXPECT_TRUE(aligned(dv1.begin(), rng::views::iota(1)).first);
  // auto x = rng::views::iota(1).begin();
  // decltype(x)::foo = 1;
  // EXPECT_TRUE(aligned(rng::views::iota(1).begin(), dv1.begin()).first);

  // May not be useful to support
  // distributed and local vector
  // EXPECT_FALSE(aligned(dv1.begin(), v1.begin()));
}
