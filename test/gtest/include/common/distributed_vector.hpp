// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class DistributedVector : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(DistributedVector);

TYPED_TEST_P(DistributedVector, Requirements) {
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

TYPED_TEST_P(DistributedVector, Constructors) {
  TypeParam a1(10);
  iota(a1, 10);

  TypeParam a3(10, 2);
  if (comm_rank == 0) {
    LocalVec<TypeParam> v3(10, 2);
    EXPECT_TRUE(equal(a3, v3));
  }
}

REGISTER_TYPED_TEST_SUITE_P(DistributedVector, Requirements, Constructors);
