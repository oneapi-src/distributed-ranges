// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DV = xhp::distributed_vector<T>;
using LV = std::vector<T>;

TEST(MhpSort, BigRandom) {
  LV v = generate_random<T>(750000, 2048);
  auto size = v.size();
  DV d_v(size);

  dr::mhp::copy(0, v, d_v.begin());
  barrier();

  std::sort(v.begin(), v.end());
  xhp::sort(d_v);

  barrier();
  EXPECT_TRUE(equal(v, d_v));
}
