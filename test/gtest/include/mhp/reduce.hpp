// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class ReduceMHP : public testing::Test {
public:
};

TYPED_TEST_SUITE(ReduceMHP, TestTypes);

TYPED_TEST(ReduceMHP, RangeRoot) {
  std::size_t root = 0;
  std::size_t n = 10;
  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result = mhp::reduce(default_policy(dv_a), dv_a, 0, std::plus{}, root);

  if (comm_rank == root) {
    LocalVec<TypeParam> a(n);
    rng::iota(a, 100);
    auto result = std::reduce(a.begin(), a.end(), 0, std::plus{});
    EXPECT_EQ(result, d_result);
  }
}

TYPED_TEST(ReduceMHP, IteratorsRoot) {
  std::size_t root = 0;
  std::size_t n = 10;
  TypeParam dv_a(n);
  iota(dv_a, 100);
  auto d_result = mhp::reduce(default_policy(dv_a), dv_a.begin() + 1,
                              dv_a.end() - 1, 0, std::plus{}, root);

  if (comm_rank == root) {
    LocalVec<TypeParam> a(n);
    rng::iota(a, 100);
    auto result = std::reduce(a.begin() + 1, a.end() - 1, 0, std::plus{});
    EXPECT_EQ(result, d_result);
  }
}
