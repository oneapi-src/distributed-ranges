// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-tests.hpp"

using T = int;
using DV = lib::distributed_vector<T>;
using V = std::vector<T>;

void check_local_span(std::size_t n) {
  int iota_base = 100;
  DV dv1(n), dv2(n), dv3(n);
  rng::iota(dv1, iota_base);
  rng::iota(dv2, iota_base);
  rng::iota(dv3, iota_base);
  dv1.fence();
  dv2.fence();
  dv3.fence();

  for (auto &e : dv1.local()) {
    e += 1000;
  }

  for (auto &e : dv2 | lib::local_span()) {
    e += 1000;
  }

  for (auto &e : dv3 | rng::views::take(6) | lib::local_span()) {
    e += 1000;
  }

  dv1.fence();
  dv2.fence();
  dv3.fence();
  if (comm_rank == 0) {
    V v1(n), v2(n), v3(n);
    rng::iota(v1, iota_base);
    rng::iota(v2, iota_base);
    rng::iota(v3, iota_base);

    for (auto &e : v1) {
      e += 1000;
    }

    for (auto &e : v3 | rng::views::take(6)) {
      e += 1000;
    }

    EXPECT_TRUE(equal(v1, dv1));
    EXPECT_TRUE(equal(v1, dv2));
    EXPECT_TRUE(equal(v3, dv3));
  }
}

TEST(CpuMpiTests, LocalSpanView) { check_local_span(10); }
