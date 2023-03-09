// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Take : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(Take);

TYPED_TEST_P(Take, DISABLED_Basic) {
  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto t = rng::views::take(dv_a, 6);
  EXPECT_TRUE(check_segments(t));
  barrier();
  xhp::for_each(default_policy(dv_a), t, neg);

  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::views::take(a, 6), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}

REGISTER_TYPED_TEST_SUITE_P(Take, DISABLED_Basic);
