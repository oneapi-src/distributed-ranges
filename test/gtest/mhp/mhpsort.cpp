// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DV = xhp::distributed_vector<T>;
using LV = std::vector<T>;

TEST(MhpSort, BigRandom) {
  LV v = generate_random<T>(2000000, 10000);
  auto size = v.size();
  DV d_v(size);
  std::cout << "BigRandom: dv size " << size << std::endl;
  dr::mhp::copy(0, v, d_v.begin());

  std::sort(v.begin(), v.end());
  dr::mhp::sort(d_v);

  EXPECT_TRUE(equal(v, d_v));
}
