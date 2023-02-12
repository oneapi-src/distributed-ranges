// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

std::size_t radius = 4;
std::size_t n = 10;

TEST(MhpTests, Stencil) {
  DV dv_a(n, mhp::stencil(radius));
  DV dv_b(n, mhp::stencil(radius));
  V v(n);

  mhp::iota(dv_a, 10);
  dv_a.halo().exchange_begin();
  dv_a.halo().exchange_finalize();

  mhp::fill(dv_b, 100);
  dv_b.halo().exchange_begin();
  dv_b.halo().exchange_finalize();

  if (comm_rank == 0) {
    fmt::print("segments(dv): {}\n", lib::ranges::segments(dv_a));
    rng::iota(v, 10);
    EXPECT_TRUE(check_segments(dv_a));
    EXPECT_TRUE(equal(v, dv_a));
  }

  MPI_Barrier(comm);

#if 0
  auto sum = [](auto &v) {
    T s = v;
    for (std::size_t i = 0; i <= radius; i++) {
      s += p[-i];
      s += p[i];
    }

    return s;
  };

  auto inner = rng::subrange(dv.begin() + radius, dv.end() - radius);
  mhp::transform(inner
  mhp::transform(dv.begin() + radius, dv.end() - radius, init_win);

  if (comm_rank == 0) {
    std::for_each(v.begin() + radius, v.end() - radius, init_win);
    EXPECT_TRUE(equal(v, dv));
  }
#endif
}
