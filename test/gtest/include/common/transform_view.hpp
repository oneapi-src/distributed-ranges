// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class TransformView : public testing::Test {
public:
};

TYPED_TEST_SUITE(TransformView, TestTypes);

TYPED_TEST(TransformView, Basic) {
  Ops1<TypeParam> ops(10);

  auto negate = [](auto v) { return -v; };
  EXPECT_TRUE(check_view(rng::views::transform(ops.vec, negate),
                         lib::views::transform(ops.dist_vec, negate)));
}
