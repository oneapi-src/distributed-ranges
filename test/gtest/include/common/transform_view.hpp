// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class TransformView : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(TransformView);

TYPED_TEST_P(TransformView, Basic) {
  using DV = typename TypeParam::DV;
  using V = typename TypeParam::V;

  std::size_t n = 10;

  auto neg = [](auto v) { return -v; };
  DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto dv_t = lib::views::transform(dv_a, neg);
  EXPECT_TRUE(check_segments(dv_t));
  barrier();
  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    auto t = rng::views::transform(a, neg);
    EXPECT_TRUE(equal(t, dv_t));
  }
}

REGISTER_TYPED_TEST_SUITE_P(TransformView, Basic);
