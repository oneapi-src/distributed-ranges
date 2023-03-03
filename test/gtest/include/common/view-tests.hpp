// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

TYPED_TEST_P(CommonTests, Subrange) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  TypeParam::for_each(TypeParam::policy(),
                      rng::subrange(dv_a.begin() + 1, dv_a.end() - 1), neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::subrange(a.begin() + 1, a.end() - 1), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}

TYPED_TEST_P(CommonTests, Drop) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  TypeParam::for_each(TypeParam::policy(), rng::views::drop(dv_a, 2), neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::views::drop(a, 2), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}

TYPED_TEST_P(CommonTests, DISABLED_Take) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  TypeParam::for_each(TypeParam::policy(), rng::views::take(dv_a, 6), neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::views::take(a, 6), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}
