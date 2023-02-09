// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

struct increment {
  auto operator()(auto &&v) const { v++; }
};

TEST(MhpTests, Subrange) {
  DV dv(10);
  auto r = rng::subrange(dv.begin(), dv.end());
  rng::segments_(r);
  static_assert(lib::distributed_range<decltype(r)>);
}

TEST(MhpTests, Zip) {
  // auto x = rng::views::iota(1);
  // static_assert(lib::distributed_contiguous_range<decltype(x)>);
  DV dv1(10), dv2(10);
  mhp::iota(dv1, 10);
  mhp::iota(dv2, 20);
  auto dzv = rng::views::zip(dv1, dv2);
  fmt::print("dzv: {}\n"
             "  dv1: {}\n"
             "  dv2: {}\n"
             "  segments(dv1): {}\n"
             "  segments(dv2): {}\n",
             dzv, dv1, dv2, lib::ranges::segments(dv1),
             lib::ranges::segments(dv2));
  static_assert(lib::is_zip_view_v<decltype(dzv)>);
  static_assert(lib::distributed_range<decltype(dzv)>);
#if 0
  auto incr_0 = [](auto &x) {
    std::get<0>(x)++;
  };
  mhp::for_each(dzv, incr_0);
  fmt::print("after foreach\n"
             "dzv: {}\n"
             "  dv1: {}\n"
             "  dv2: {}\n",
             dzv, dv1, dv2);
#endif
}

TEST(MhpTests, Take) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  auto aview = rng::views::take(a, 2);
  auto dv_aview = rng::views::take(dv_a, 2);

  mhp::iota(dv_a, 20);
  if (comm == 0) {
    rng::iota(a, 20);
    EXPECT_TRUE(equal(aview, dv_aview));
  }

  mhp::for_each(dv_aview, increment{});
  if (comm == 0) {
    rng::for_each(aview, increment{});
    EXPECT_TRUE(equal(aview, dv_aview));
  }
}

TEST(MhpTests, Drop) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  auto aview = rng::views::drop(a, 2);
  auto dv_aview = rng::views::drop(dv_a, 2);

  mhp::iota(dv_a, 20);
  if (comm == 0) {
    rng::iota(a, 20);
    EXPECT_TRUE(equal(aview, dv_aview));
  }

  mhp::for_each(dv_aview, increment{});
  if (comm == 0) {
    rng::for_each(aview, increment{});
    EXPECT_TRUE(equal(aview, dv_aview));
  }
}
