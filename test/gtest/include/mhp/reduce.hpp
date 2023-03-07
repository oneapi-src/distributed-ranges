// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class ReduceMHP : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(ReduceMHP);

TYPED_TEST_P(ReduceMHP, RangeRoot) {
  std::size_t root = 0;
  std::size_t n = 10;
  typename TypeParam::DV dv_a(n);
  mhp::iota(dv_a, 100);
  auto d_result = mhp::reduce(TypeParam::policy(), dv_a, 0, std::plus{}, root);

  if (comm_rank == root) {
    typename TypeParam::V a(n);
    rng::iota(a, 100);
    auto result = std::reduce(a.begin(), a.end(), 0, std::plus{});
    EXPECT_EQ(result, d_result);
  }
}

TYPED_TEST_P(ReduceMHP, IteratorsRoot) {
  std::size_t root = 0;
  std::size_t n = 10;
  typename TypeParam::DV dv_a(n);
  mhp::iota(dv_a, 100);
  auto d_result = mhp::reduce(TypeParam::policy(), dv_a.begin() + 1,
                              dv_a.end() - 1, 0, std::plus{}, root);

  if (comm_rank == root) {
    typename TypeParam::V a(n);
    rng::iota(a, 100);
    auto result = std::reduce(a.begin() + 1, a.end() - 1, 0, std::plus{});
    EXPECT_EQ(result, d_result);
  }
}

REGISTER_TYPED_TEST_SUITE_P(ReduceMHP, RangeRoot, IteratorsRoot);
