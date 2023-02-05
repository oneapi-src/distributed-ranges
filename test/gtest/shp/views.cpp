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
