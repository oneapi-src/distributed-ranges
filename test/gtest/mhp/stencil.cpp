// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = dr::mhp::distributed_vector<T>;
using DVI = typename DV::iterator;

std::size_t radius = 4;
std::size_t n = 10;

TEST(MhpTests, Stencil) {
  DV dv_in(n, dr::halo_bounds(radius));
  DV dv_out(n, dr::halo_bounds(radius));
  V v_in(n);

  dr::mhp::iota(dv_in, 10);
  dv_in.halo().exchange();

  dr::mhp::fill(dv_out, 100);
  dv_out.halo().exchange();

  if (comm_rank == 0) {
    rng::iota(v_in, 10);
    EXPECT_TRUE(check_segments(dv_in));
    EXPECT_TRUE(equal(v_in, dv_in));
  }

  MPI_Barrier(comm);

  auto sum = [](auto &&v) {
    T s = v;
    auto p = &v;
    for (std::size_t i = 0; i <= radius; i++) {
      s += p[-i];
      s += p[i];
    }

    return s;
  };

  dr::mhp::transform(dv_in.begin() + radius, dv_in.end() - radius,
                     dv_out.begin() + radius, sum);

  if (comm_rank == 0) {
    V v_out(n);
    rng::fill(v_out, 100);
    std::transform(v_in.begin() + radius, v_in.end() - radius,
                   v_out.begin() + radius, sum);
    EXPECT_TRUE(check_unary_op(v_in, v_out, dv_out));
  }
}
