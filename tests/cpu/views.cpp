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

TEST(CpuMpiTests, ZipViewLocal) {
  const int n = 10;
  V a(n), b(n);
  rng::iota(a, 100);
  rng::iota(b, 1000);

  auto zv = lib::zip_view(a, b);
  using ZV = decltype(zv);
  // using I = decltype(zv.begin());

  auto szv = rng::views::zip(a, b);
  using SZV = decltype(szv);

  auto it = zv.begin();
  auto sit = szv.begin();
  ++it;
  ++sit;
  EXPECT_EQ(*sit, *it);
  it++;
  sit++;
  EXPECT_EQ(*sit, *it);

  EXPECT_EQ(a[0], *begin(a));
  EXPECT_EQ(*begin(szv), *begin(zv));
  auto index = 2;
  EXPECT_EQ(szv.begin()[index], zv.begin()[index]);
  static_assert(rng::input_range<ZV>);
  static_assert(rng::input_range<SZV>);
  static_assert(
      std::indirectly_comparable<decltype(zv.begin()), decltype(szv.begin()),
                                 rng::equal_to>);
  EXPECT_TRUE(equal(szv, zv));

  auto [e1, e2] = *begin(szv);
  auto [f1, f2] = *begin(zv);
  e1 = f1 = 22;
  e2 = f2 = 23;
  EXPECT_TRUE(equal(szv, zv));

  auto iota_1 = rng::views::iota(1);

  auto szi = rng::views::zip(a, iota_1);
  auto zi = lib::zip_view(a, iota_1);
  EXPECT_TRUE(equal(szi, zi));

  auto sz3i = rng::views::zip(a, b, iota_1);
  auto z3i = lib::zip_view(a, b, iota_1);
  EXPECT_TRUE(equal(sz3i, z3i));

  V sci(n), ci(n);
  auto iota = rng::views::iota(1);
  auto szi2 = rng::views::zip(a, iota, sci);
  rng::for_each(szi2, add_2{});
  auto zi2 = lib::zip_view(a, iota, ci);
  rng::for_each(zi2, add_2{});
  EXPECT_TRUE(binary_check(a, iota | rng::views::take(a.size()), sci, ci));
}

TEST(CpuMpiTests, ZipViewDistributed) {
  const int n = 10;
  V a(n), b(n), c(n), ci(n);
  DV dv_a(n), dv_b(n), dv_c1(n), dv_c2(n), dv_ci(n), dv_cri(n);

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

  auto &&dv_z2 = rng::views::zip(dv_a, dv_b, dv_c2);
  lib::for_each(dv_z2, add_2{});
  EXPECT_TRUE(binary_check(a, b, c, dv_c2));

  auto iota = rng::views::iota(1);
  auto zi = rng::views::zip(a, iota, ci);
  rng::for_each(zi, add_2{});
  auto dv_zi = lib::zip_view(dv_a, iota, dv_ci);
  rng::for_each(dv_zi, add_2{});
  EXPECT_TRUE(binary_check(a, iota | rng::views::take(a.size()), ci, dv_ci));
}
