// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using V = std::vector<int>;
using CV = const std::vector<int>;
using CVR = const std::vector<int> &;

using DV = shp::distributed_vector<int>;
using CDV = const shp::distributed_vector<int>;
using CDVR = const shp::distributed_vector<int> &;

TEST(ShpTests, DistributedVector) {
  static_assert(rng::random_access_range<V>);
  static_assert(rng::random_access_range<CV>);
  static_assert(rng::random_access_range<CVR>);

  static_assert(rng::random_access_range<DV>);
  static_assert(rng::random_access_range<CDV>);
  static_assert(rng::random_access_range<CDVR>);
}

TEST(ShpTests, DistributedVectorSegments) {
  const int n = 10;
  DV dv_a(n);
  std::iota(dv_a.begin(), dv_a.end(), 20);

  auto second = dv_a.begin() + 2;
  EXPECT_EQ(second[0], lib::ranges::segments(second)[0][0]);
}

TEST(ShpTests, DistributedVectorConstructorFill) {
  const int n = 10;
  const int value = 1;
  DV dv_a(n, value);
  DV dv_b = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  EXPECT_TRUE(std::equal(dv_a.begin(), dv_a.end(), dv_b.begin()));
}

TEST(ShpTest, DistributedVectorConstructorInitializerList) {
  const int n = 10;
  DV dv_a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  DV dv_b(n);
  std::iota(dv_b.begin(), dv_b.end(), 0);

  EXPECT_TRUE(std::equal(dv_a.begin(), dv_a.end(), dv_b.begin()));
}
