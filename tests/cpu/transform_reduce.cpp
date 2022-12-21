// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-tests.hpp"

void check_transform_reduce(std::string title, std::size_t n, std::size_t b,
                            std::size_t e) {
  auto reduce_op = std::plus<>();
  auto transform_op = [](auto &&v) { return v * v; };
  int init = 10000;
  int root = 0;
  lib::drlog.debug("{}\n", title);

  int iota_base = 100;

  lib::distributed_vector<int> dvi1(n);
  rng::iota(dvi1, iota_base);
  dvi1.fence();

  auto dval1 = lib::transform_reduce(root, dvi1.begin() + b, dvi1.begin() + e,
                                     init, reduce_op, transform_op);

  lib::distributed_vector<int> dvi2(n);
  rng::iota(dvi2, iota_base);
  dvi2.fence();

  if (comm_rank == root) {
    auto dval2 = std::transform_reduce(dvi2.begin() + b, dvi2.begin() + e, init,
                                       reduce_op, transform_op);

    std::vector<int> v(n);
    rng::iota(v, iota_base);
    auto val = std::transform_reduce(v.begin() + b, v.begin() + e, init,
                                     reduce_op, transform_op);
    EXPECT_EQ(val, dval1);
    EXPECT_EQ(val, dval2);
  }
}

TEST(CpuMpiTests, TransformReduceDistributedVector) {
  std::size_t n = 10;

  check_transform_reduce("full vector", n, 0, n);
  check_transform_reduce("partial vector", n, n / 2 - 1, n / 2 + 1);
}
