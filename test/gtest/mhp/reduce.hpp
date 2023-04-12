// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class ReduceMHP : public testing::Test {
public:
};

TYPED_TEST_SUITE(ReduceMHP, AllTypes);

const std::size_t root = 0;

TYPED_TEST(ReduceMHP, RootRange) {
  Ops1<TypeParam> ops(10);

  auto result = dr::mhp::reduce(default_policy(ops.dist_vec), ops.dist_vec, 0,
                                std::plus{}, root);

  if (comm_rank == root) {
    EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 0, std::plus{}),
              result);
  }
}

TYPED_TEST(ReduceMHP, RootIterators) {
  Ops1<TypeParam> ops(10);

  auto result =
      dr::mhp::reduce(default_policy(ops.dist_vec), ops.dist_vec.begin() + 1,
                      ops.dist_vec.end() - 1, 0, std::plus{}, root);

  if (comm_rank == root) {
    EXPECT_EQ(
        std::reduce(ops.vec.begin() + 1, ops.vec.end() - 1, 0, std::plus{}),
        result);
  }
}
