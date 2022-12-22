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

struct add_2 {
  void operator()(auto &&z) {
    auto [a, b, c] = z;
    c = a + b;
  }
};

TEST(CpuMpiTests, ZipView) {
  const int n = 10;
  V a(n), b(n), c(n);
  DV dv_a(n), dv_b(n), dv_c1(n), dv_c2(n);

  rng::iota(a, 100);
  rng::iota(b, 1000);
  lib::iota(dv_a, 100);
  lib::iota(dv_b, 1000);

  auto &&z = rng::views::zip(a, b, c);
  rng::for_each(z, add_2{});

  auto &&dv_z1 = rng::views::zip(dv_a, dv_b, dv_c1);
  rng::for_each(dv_z1, add_2{});
  dv_c1.fence();
  EXPECT_TRUE(binary_check(a, b, c, dv_c1));

#if 0
  auto &&dv_z2 = rng::views::zip(dv_a, dv_b, dv_c2);
  lib::for_each(dv_z2, add_2{});
  dv_c2.fence();
  EXPECT_TRUE(binary_check(a, b, c, dv_c2));
#endif
}
