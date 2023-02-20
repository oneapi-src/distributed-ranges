// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = typename DV::iterator;

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
  DV dv1(10), dv2(10);
  mhp::iota(dv1, 10);
  mhp::iota(dv2, 20);
  auto dzv = rng::views::zip(dv1, dv2);
  static_assert(lib::is_zip_view_v<decltype(dzv)>);
  static_assert(lib::distributed_range<decltype(dzv)>);
  EXPECT_TRUE(check_segments(dzv));
  EXPECT_TRUE(check_segments(dzv.begin()));

  dv1.barrier();
  auto incr_first = [](auto x) { x.first++; };
  mhp::for_each(dzv, incr_first);

  if (comm_rank == 0) {
    V v1(10), v2(10);
    rng::iota(v1, 10);
    rng::iota(v2, 20);
    auto zv = rng::views::zip(v1, v2);
    rng::for_each(zv, incr_first);

    EXPECT_TRUE(equal(zv, dzv));
  }
}

TEST(MhpTests, Take) {
  const int n = 10;
  DV dv_a(n);
  mhp::iota(dv_a, 20);

  auto dv_aview = rng::views::take(dv_a, 2);
  EXPECT_TRUE(check_segments(dv_aview));

  if (comm == 0) {
    V a(n);
    auto aview = rng::views::take(a, 2);
    rng::iota(a, 20);
    EXPECT_TRUE(equal(aview, dv_aview));
  }

  dv_a.barrier();
  mhp::for_each(dv_aview, increment{});

  if (comm == 0) {
    V a(n);
    auto aview = rng::views::take(a, 2);
    rng::iota(a, 20);
    rng::for_each(aview, increment{});
    EXPECT_TRUE(equal(aview, dv_aview));
  }
}

TEST(MhpTests, Drop) {
  const int n = 10;

  DV dv_a(n);
  mhp::iota(dv_a, 20);
  auto dv_aview = rng::views::drop(dv_a, 2);

  EXPECT_TRUE(check_segments(dv_aview));
  if (comm == 0) {
    V a(n);
    rng::iota(a, 20);
    auto aview = rng::views::drop(a, 2);
    EXPECT_TRUE(equal(aview, dv_aview));
  }

  dv_a.barrier();
  mhp::for_each(dv_aview, increment{});

  if (comm == 0) {
    V a(n);
    rng::iota(a, 20);
    auto aview = rng::views::drop(a, 2);
    rng::for_each(aview, increment{});
    EXPECT_TRUE(equal(aview, dv_aview));
  }
}

#if 0
TEST(MhpTests, Transform) {
  const int n = 10;

  DV dv_a(n);
  mhp::iota(dv_a, 20);
  auto incr = [](auto x) { return x + 1; };
  auto dv_a_view = lib::views::transform(dv_a, incr);

  if (comm == 0) {
    V v_a(n);
    rng::iota(v_a, 20);
    auto v_a_view = rng::views::transform(v_a, incr);
    EXPECT_TRUE(equal(v_a_view, dv_a_view));
  }
}
#endif
