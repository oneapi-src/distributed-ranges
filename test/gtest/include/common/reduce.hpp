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

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result = xhp::reduce(default_policy(dv_a), dv_a, 0, std::plus{});

  LocalVec<TypeParam> a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin(), a.end(), 0, std::plus{});
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, Iterators) {
  std::size_t n = 10;

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result = xhp::reduce(default_policy(dv_a), dv_a.begin() + 1,
                              dv_a.end() - 1, 0, std::plus{});

  LocalVec<TypeParam> a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin() + 1, a.end() - 1, 0, std::plus{});
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, IteratorsDefaultOp) {
  std::size_t n = 10;

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result =
      xhp::reduce(default_policy(dv_a), dv_a.begin() + 1, dv_a.end() - 1, 0);

  LocalVec<TypeParam> a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin() + 1, a.end() - 1, 0);
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, IteratorsDefaultInit) {
  std::size_t n = 10;

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result =
      xhp::reduce(default_policy(dv_a), dv_a.begin() + 1, dv_a.end() - 1);

  LocalVec<TypeParam> a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin() + 1, a.end() - 1);
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, RangeDefaultOp) {
  std::size_t n = 10;

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result = xhp::reduce(default_policy(dv_a), dv_a, 0);

  LocalVec<TypeParam> a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin(), a.end(), 0);
  EXPECT_EQ(result, d_result);
}

TYPED_TEST_P(Reduce, RangeDefaultInit) {
  std::size_t n = 10;

  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result = reduce(default_policy(dv_a), dv_a);

  LocalVec<TypeParam> a(n);
  rng::iota(a, 100);
  auto result = std::reduce(a.begin(), a.end());
  EXPECT_EQ(result, d_result);
}

REGISTER_TYPED_TEST_SUITE_P(Reduce, Range, RangeDefaultOp, RangeDefaultInit,
                            Iterators, IteratorsDefaultOp,
                            IteratorsDefaultInit);
