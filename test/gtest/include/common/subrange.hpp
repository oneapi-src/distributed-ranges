// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Subrange : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(Subrange);

TYPED_TEST_P(Subrange, Basic) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto sr = rng::subrange(dv_a.begin() + 1, dv_a.end() - 1);
  EXPECT_TRUE(check_segments(sr));
  barrier();
  xhp::for_each(TypeParam::policy(), sr, neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::subrange(a.begin() + 1, a.end() - 1), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}

REGISTER_TYPED_TEST_SUITE_P(Subrange, Basic);
