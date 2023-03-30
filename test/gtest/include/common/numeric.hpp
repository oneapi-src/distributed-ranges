// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename AllocT> class NumericTest : public testing::Test {
public:
  //   using DistVec = xhp::distributed_vector<typename AllocT::value_type,
  //   AllocT>; using LocalVec = std::vector<typename AllocT::value_type>;
};

TYPED_TEST_SUITE(NumericTest, AllTypes);

TYPED_TEST(NumericTest, IotaRange) {
  Ops1<TypeParam> ops(10);
  auto input = ops.vec;
  xhp::iota(ops.dist_vec, 1);
  rng::iota(ops.vec, 1);
  EXPECT_TRUE(check_unary_op(input, ops.vec, ops.dist_vec));
}
