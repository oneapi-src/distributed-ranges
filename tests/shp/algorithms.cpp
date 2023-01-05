// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using T = int;
using DV = shp::distributed_vector<T, shp::shared_allocator<T>>;
using V = std::vector<T>;

TEST(ShpTests, Iota) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  std::iota(a.begin(), a.end(), 20);
  std::iota(dv_a.begin(), dv_a.end(), 20);
  EXPECT_TRUE(equal(a, dv_a));
}
