// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class DistributedVectorTestTypes : public testing::Test {
public:
};

TYPED_TEST_SUITE(DistributedVectorTestTypes, TestTypes);

TYPED_TEST(DistributedVectorTestTypes, Requirements) {
  TypeParam dv(10);

  static_assert(rng::random_access_range<decltype(dv.segments())>);
  static_assert(rng::random_access_range<decltype(dv.segments()[0])>);
  static_assert(rng::viewable_range<decltype(dv.segments())>);

  static_assert(std::forward_iterator<decltype(dv.begin())>);
  static_assert(lib::distributed_iterator<decltype(dv.begin())>);

  static_assert(rng::forward_range<decltype(dv)>);
  static_assert(rng::random_access_range<decltype(dv)>);
  static_assert(lib::distributed_contiguous_range<decltype(dv)>);
}

// gtest support
TYPED_TEST(DistributedVectorTestTypes, Stream) {
  Ops1<TypeParam> ops(10);
  std::cout << ops.dist_vec << "\n";
}

// gtest support
TYPED_TEST(DistributedVectorTestTypes, Equality) {
  Ops1<TypeParam> ops(10);
  iota(ops.dist_vec, 100);
  rng::iota(ops.vec, 100);
  EXPECT_TRUE(ops.dist_vec == ops.vec);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TEST(DistributedVector, ConstructorBasic) {
  xhp::distributed_vector<int> dist_vec(10);
  iota(dist_vec, 100);

  std::vector<int> local_vec(10);
  rng::iota(local_vec, 100);

  EXPECT_EQ(local_vec, dist_vec);
}

TEST(DistributedVector, ConstructorFill) {
  xhp::distributed_vector<int> dist_vec(10, 1);

  std::vector<int> local_vec(10, 1);

  EXPECT_EQ(local_vec, dist_vec);
}

TEST(DistributedVector, ConstructorBasicAOS) {
  OpsAOS ops(10);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}

TEST(DistributedVector, ConstructorFillAOS) {
  OpsAOS::Struct fill_value{1, 2};
  OpsAOS::dist_vec_type dist_vec(10, fill_value);
  OpsAOS::vec_type local_vec(10, fill_value);

  EXPECT_EQ(local_vec, dist_vec);
}
