// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

TYPED_TEST_P(CommonTests, ForEach) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  xhp::for_each(TypeParam::policy(), dv_a, neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(a, neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}
