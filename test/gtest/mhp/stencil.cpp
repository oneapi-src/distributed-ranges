// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

std::size_t radius = 4;
std::size_t n = 10 + 2 * radius;

TEST(MhpTests, Stencil) {
  mhp::stencil stencil((mhp::stencil::bounds(radius)));
  DV dv(n, stencil);
  V v(n);

  mhp::iota(dv, 10);

  if (comm_rank == 0) {
    fmt::print("segments(dv): {}\n", lib::ranges::segments(dv));
    rng::iota(v, 10);
    EXPECT_TRUE(check_segments(dv));
    EXPECT_TRUE(equal(v, dv));
  }

  dv.barrier();

  auto init_win = [](auto &v) {
    auto p = &v;
    for (std::size_t i = 0; i < radius; i++) {
      p[-i] = 1;
      p[i] = 1;
    }
  };

  mhp::for_each(dv.begin() + radius, dv.end() - radius, init_win);

  if (comm_rank == 0) {
    std::for_each(v.begin() + radius, v.end() - radius, init_win);
    EXPECT_TRUE(equal(v, dv));
  }
}
