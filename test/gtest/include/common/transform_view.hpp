// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class TransformView : public testing::Test {
public:
};

TYPED_TEST_SUITE(TransformView, AllTypes);

TYPED_TEST(TransformView, Basic) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  EXPECT_TRUE(check_view(rng::views::transform(ops.vec, negate),
                         lib::views::transform(ops.dist_vec, negate)));
}

TYPED_TEST(TransformView, All) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  EXPECT_TRUE(
      check_view(rng::views::transform(rng::views::all(ops.vec), negate),
                 lib::views::transform(rng::views::all(ops.dist_vec), negate)));
}

TYPED_TEST(TransformView, Move) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto l_value = rng::views::all(ops.vec);
  auto dist_l_value = rng::views::all(ops.dist_vec);
  EXPECT_TRUE(
      check_view(rng::views::transform(std::move(l_value), negate),
                 lib::views::transform(std::move(dist_l_value), negate)));
}

TYPED_TEST(TransformView, L_Value) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  auto l_value = rng::views::all(ops.vec);
  auto dist_l_value = rng::views::all(ops.dist_vec);
  EXPECT_TRUE(check_view(rng::views::transform(l_value, negate),
                         lib::views::transform(dist_l_value, negate)));
}
