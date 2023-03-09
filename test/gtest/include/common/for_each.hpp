// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class ForEach : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(ForEach);

TYPED_TEST_P(ForEach, Basic) {
  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  TypeParam dv_a(n);
  iota(dv_a, 100);
  xhp::for_each(default_policy(dv_a), dv_a, neg);

  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(a, neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}

REGISTER_TYPED_TEST_SUITE_P(ForEach, Basic);
