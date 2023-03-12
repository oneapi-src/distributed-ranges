// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class TransformView : public testing::Test {
public:
};

TYPED_TEST_SUITE(TransformView, TestTypes);

TYPED_TEST(TransformView, Basic) {
  std::size_t n = 10;

  auto neg = [](auto v) { return -v; };
  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto dv_t = lib::views::transform(dv_a, neg);
  EXPECT_TRUE(check_segments(dv_t));
  barrier();
  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    auto t = rng::views::transform(a, neg);
    EXPECT_TRUE(equal(t, dv_t));
  }
}
