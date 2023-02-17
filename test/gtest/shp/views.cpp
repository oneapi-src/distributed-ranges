// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using T = int;
using DV = shp::distributed_vector<T, shp::shared_allocator<T>>;
using V = std::vector<T>;

struct increment {
  auto operator()(auto &&v) const { v++; }
};

TEST(ShpTests, Take) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  std::iota(a.begin(), a.end(), 20);
  std::iota(dv_a.begin(), dv_a.end(), 20);

  auto aview = a | rng::views::take(2);
  auto dv_aview = dv_a | rng::views::take(2);
  EXPECT_TRUE(equal(aview, dv_aview));

  std::ranges::for_each(aview, increment{});
  shp::for_each(shp::par_unseq, dv_a, increment{});
  EXPECT_TRUE(equal(aview, dv_aview));
}

template <lib::distributed_range R> void dr(R &&) {}

template <lib::remote_range R> void rr(R &&) {}

TEST(ShpTests, Zip) {
  const int n = 10;
  DV dv_a(n), dv_b(n);
  rng::iota(dv_a, 100);
  rng::iota(dv_b, 200);
  auto dz = shp::views::zip(dv_a, dv_b, dv_a);
  auto dz2 = shp::views::zip(dv_a, dv_b);
  auto dzi = shp::views::zip(rng::views::iota(1, 10), dv_b, dv_a);

  DV v_a(n), v_b(n);
  rng::iota(v_a, 100);
  rng::iota(v_b, 200);
  auto z = rng::views::zip(v_a, v_b, v_a);
  auto z2 = rng::views::zip(v_a, v_b);
  auto zi = rng::views::zip(rng::views::iota(1, 10), dv_b, dv_a);

  EXPECT_TRUE(equal(z, dz));
  EXPECT_TRUE(equal(zi, dzi));

#if 0
  // zip and dr zip have different value types: tuple/pair
  EXPECT_TRUE(equal(z2, dz2));
#endif

  fmt::print("a: {}\n"
             "b: {}\n"
             "dz: {}\n"
             "dzi: {}\n"
             "segments(dz): {}\n"
             "segments(dzi): {}\n"
             "z: {}\n",
             dv_a, dv_b, dz, dzi, lib::ranges::segments(dz),
             lib::ranges::segments(dzi), z);
}

TEST(ShpTests, Drop) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  auto incr = [](auto &&v) { v++; };

  std::iota(a.begin(), a.end(), 20);
  std::iota(dv_a.begin(), dv_a.end(), 20);

  auto aview = a | rng::views::drop(2);
  auto dv_aview = dv_a | rng::views::drop(2);
  EXPECT_TRUE(equal(aview, dv_aview));

  fmt::print("segments(dv_a):     {}\n"
             "segments(dv_aview): {}\n"
             //"xsegments(dv_aview): {}\n"
             ,
             lib::ranges::segments(dv_a), lib::ranges::segments(dv_aview)
             // ranges::xsegments_(dv_aview)
  );
  std::ranges::for_each(aview, incr);
  shp::for_each(shp::par_unseq, dv_aview, incr);
  fmt::print("segments(dv_a):     {}\n"
             "segments(dv_aview): {}\n",
             lib::ranges::segments(dv_a), lib::ranges::segments(dv_aview));
  EXPECT_TRUE(equal(aview, dv_aview));
}
