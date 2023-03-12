// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Subrange : public testing::Test {
public:
};

TYPED_TEST_SUITE(Subrange, TestTypes);

TYPED_TEST(Subrange, Basic) {
  std::size_t n = 10;

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto sr = rng::subrange(dv_a.begin() + 1, dv_a.end() - 1);
  EXPECT_TRUE(check_segments(sr));

  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n);
    rng::iota(a, 100);
    EXPECT_TRUE(equal(rng::subrange(a.begin() + 1, a.end() - 1), sr));
  }
}

TYPED_TEST(Subrange, ForEach) {
  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto sr = rng::subrange(dv_a.begin() + 1, dv_a.end() - 1);
  EXPECT_TRUE(check_segments(sr));
  barrier();
  xhp::for_each(default_policy(dv_a), sr, neg);

  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(rng::subrange(a.begin() + 1, a.end() - 1), neg);
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}
