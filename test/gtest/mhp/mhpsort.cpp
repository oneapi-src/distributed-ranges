// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// TODO: add sort tests with ISHMEM, currently doesn't compile
using T = int;
using DV = xhp::distributed_vector<T>;
using LV = std::vector<T>;

// disabled until the issue with Intel MPI is solved
// https://github.com/orgs/oneapi-src/projects/15/views/2?pane=issue&itemId=38871430
TEST(MhpSort, DISABLED_BigRandom) {
  LV v = generate_random<T>(2000000, 10000);
  auto size = v.size();
  DV d_v(size);
  std::cout << "BigRandom: dv size " << size << std::endl;
  dr::mhp::copy(0, v, d_v.begin());

  std::sort(v.begin(), v.end());
  dr::mhp::sort(d_v);

  EXPECT_TRUE(equal(v, d_v));
}
