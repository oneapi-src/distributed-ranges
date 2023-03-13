// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = typename DV::iterator;

TEST(MhpTests, Copy) {
  std::size_t n = 10;

  DV dv_src(n), dv_dst1(n), dv_dst2(n), dv_dst3(n);
  mhp::iota(dv_src, 100);
  mhp::iota(dv_dst1, 200);
  mhp::iota(dv_dst2, 200);
  mhp::iota(dv_dst3, 200);
  mhp::copy(dv_src, dv_dst1.begin());
  mhp::copy(dv_src.begin(), dv_src.end(), dv_dst2.begin());
  mhp::copy(dv_src.begin() + 1, dv_src.end() - 1, dv_dst3.begin() + 2);

  if (comm_rank == 0) {
    V v_src(n), v_dst(n), v_dst3(n);
    rng::iota(v_src, 100);
    rng::iota(v_dst, 200);
    rng::iota(v_dst3, 200);
    rng::copy(v_src, v_dst.begin());
    EXPECT_TRUE(equal(dv_dst1, v_dst));
    EXPECT_TRUE(equal(dv_dst2, v_dst));

    std::copy(v_src.begin() + 1, v_src.end() - 1, v_dst3.begin() + 2);
    EXPECT_TRUE(equal(dv_dst3, v_dst3));
  }
}
