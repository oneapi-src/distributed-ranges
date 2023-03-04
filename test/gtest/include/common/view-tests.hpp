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
  auto sr = rng::subrange(dv_a.begin() + 1, dv_a.end() - 1);
  EXPECT_TRUE(check_segments(sr));
  TypeParam::for_each(TypeParam::policy(), sr, neg);

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
  auto d = rng::views::drop(dv_a, 2);
  EXPECT_TRUE(check_segments(d));
  TypeParam::for_each(TypeParam::policy(), d, neg);

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
  auto t = rng::views::take(dv_a, 6);
  EXPECT_TRUE(check_segments(t));
  TypeParam::for_each(TypeParam::policy(), t, neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::views::take(a, 6), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}

TYPED_TEST_P(CommonTests, TransformView) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto v) { return -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto dv_t = lib::views::transform(dv_a, neg);
  EXPECT_TRUE(check_segments(dv_t));
  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    auto t = rng::views::transform(a, neg);
    EXPECT_TRUE(equal(t, dv_t));
  }
}

TYPED_TEST_P(CommonTests, ZipView) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  DV dv_a(n), dv_b(n), dv_c(n);
  TypeParam::iota(dv_a, 100);
  TypeParam::iota(dv_b, 200);
  TypeParam::iota(dv_c, 300);

  // DISABLE 2 zip
  // auto d_z2 = zhp::zip(dv_a, dv_b);
  // EXPECT_TRUE(check_segments(d_z2));

  auto d_z3 = zhp::zip(dv_a, dv_b, dv_c);
  EXPECT_TRUE(check_segments(d_z3));

  if (comm_rank == 0) {
    V a(n), b(n), c(n);
    rng::iota(a, 100);
    rng::iota(b, 200);
    rng::iota(c, 300);

    // DISABLE 2 zip
    // auto z2 = rng::views::zip(a,  b);
    // EXPECT_TRUE(equal(z2, d_z2));

    auto z3 = rng::views::zip(a, b, c);
    EXPECT_TRUE(equal(z3, d_z3));
  }
}
