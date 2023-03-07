// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Reduce : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(Reduce);

TYPED_TEST_P(Reduce, Range) {
  std::size_t n = 10;

  typename TypeParam::DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto d_result = xhp::reduce(TypeParam::policy(), dv_a, 0, std::plus{});

  typename TypeParam::V a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin(), a.end(), 0, std::plus{});
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, Iterators) {
  std::size_t n = 10;

  typename TypeParam::DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto d_result = xhp::reduce(TypeParam::policy(), dv_a.begin() + 1,
                              dv_a.end() - 1, 0, std::plus{});

  typename TypeParam::V a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin() + 1, a.end() - 1, 0, std::plus{});
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, DefaultOp) {
  std::size_t n = 10;

  typename TypeParam::DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto d_result =
      xhp::reduce(TypeParam::policy(), dv_a.begin() + 1, dv_a.end() - 1, 0);

  typename TypeParam::V a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin() + 1, a.end() - 1, 0);
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, DefaultInit) {
  std::size_t n = 10;

  typename TypeParam::DV dv_a(n);
  TypeParam::iota(dv_a, 100);
  auto d_result =
      xhp::reduce(TypeParam::policy(), dv_a.begin() + 1, dv_a.end() - 1);

  typename TypeParam::V a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin() + 1, a.end() - 1);
  EXPECT_EQ(result, d_result);
}

REGISTER_TYPED_TEST_SUITE_P(Reduce, Range, Iterators, DefaultOp, DefaultInit);
