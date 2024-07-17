// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture

template <typename T> class ReduceMHP : public testing::Test {
public:
};

TYPED_TEST_SUITE(ReduceMHP, AllTypes);

const std::size_t root = 0;

TYPED_TEST(ReduceMHP, RootRange) {
  Ops1<TypeParam> ops(10);

  auto result = dr::mhp::reduce(root, ops.dist_vec, 0, std::plus{});

  if (comm_rank == root) {
    EXPECT_EQ(std::reduce(ops.vec.begin(), ops.vec.end(), 0, std::plus{}),
              result);
  }
}

TYPED_TEST(ReduceMHP, RootIterators) {
  Ops1<TypeParam> ops(10);

  auto result = dr::mhp::reduce(root, ops.dist_vec.begin() + 1,
                                ops.dist_vec.end() - 1, 0, std::plus{});

  if (comm_rank == root) {
    EXPECT_EQ(
        std::reduce(ops.vec.begin() + 1, ops.vec.end() - 1, 0, std::plus{}),
        result);
  }
}

// Example of code that should be compiling, but does not, described in issue DRA-192
// TYPED_TEST(ReduceMHP, NotCompiling) {
//   dr::mhp::distributed_vector<int> r1(10);

//   auto add = [](auto &&elem) {
//     return elem + 1;
//   };
  
//   auto added = dr::mhp::views::transform(r1, add);
//   auto min = [](double x, double y) { return std::min(x, y); };
//   auto result = dr::mhp::reduce(root, added, 1, min);
//   EXPECT_EQ(result, 1);
// }