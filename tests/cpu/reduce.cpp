// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-tests.hpp"

void check_reduce(std::size_t n, std::size_t b, std::size_t e) {
  auto op = std::plus<>();
  int init = 10000;
  int iota_base = 100;
  int root = 0;

  lib::distributed_vector<int> dv1(n);
  rng::iota(dv1, iota_base);
  dv1.fence();
  auto dv1_sum = lib::reduce(root, dv1.begin() + b, dv1.begin() + e, init, op);

  lib::distributed_vector<int> dv2(n);
  rng::iota(dv2, iota_base);
  dv2.fence();

  if (comm_rank == 0) {
    auto dv2_sum = std::reduce(dv2.begin() + b, dv2.begin() + e, init, op);

    std::vector<int> v(n);
    rng::iota(v, iota_base);
    auto v_sum = std::reduce(v.begin() + b, v.begin() + e, init, op);
    EXPECT_EQ(v_sum, dv1_sum);
    EXPECT_EQ(v_sum, dv2_sum);
  }
}

TEST(CpuMpiTests, ReduceDistributedVector) {
  std::size_t n = 10;

  check_reduce(n, 0, n);
  check_reduce(n, n / 2 - 1, n / 2 + 1);
}
