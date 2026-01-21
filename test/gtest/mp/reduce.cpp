// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

// Fixture

template <typename T> class ReduceMP : public testing::Test { public: };

TYPED_TEST_SUITE(ReduceMP, AllTypes);

const std::size_t root = 0;

TYPED_TEST(ReduceMP, RootRange) {
  Ops1<TypeParam> ops(10);

  auto result = dr::mp::reduce(root, ops.dist_vec, 0, std::plus{});

  if (comm_rank == root) {
    EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 0, std::plus{}),
              result);
  }
}

TYPED_TEST(ReduceMP, RootIterators) {
  Ops1<TypeParam> ops(10);

  auto result = dr::mp::reduce(root, ops.dist_vec.begin() + 1,
                               ops.dist_vec.end() - 1, 0, std::plus{});

  if (comm_rank == root) {
    EXPECT_EQ(
        std::reduce(ops.vec.begin() + 1, ops.vec.end() - 1, 0, std::plus{}),
        result);
  }
}

TYPED_TEST(ReduceMP, TransformReduce) {
  Ops1<TypeParam> ops(10);

  auto add = [](auto &&elem) { return elem + 1; };

  auto added = dr::mp::views::transform(ops.dist_vec, add);
  auto min = [](double x, double y) { return std::min(x, y); };
  auto result = dr::mp::reduce(root, added, 1, min);
  if (comm_rank == root) {
    EXPECT_EQ(result, 1);
  }
}
