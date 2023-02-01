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
