// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class DistributedVector : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(DistributedVector);

TYPED_TEST_P(DistributedVector, Requirements) {
  using DV = typename TypeParam::DV;
  using DVI = typename DV::iterator;
  DV dv(10);

  static_assert(rng::random_access_range<decltype(dv.segments())>);
  static_assert(rng::random_access_range<decltype(dv.segments()[0])>);
  static_assert(rng::viewable_range<decltype(dv.segments())>);
  static_assert(std::forward_iterator<DVI>);
  static_assert(rng::forward_range<DV>);
  static_assert(rng::random_access_range<DV>);

  static_assert(lib::distributed_iterator<decltype(dv.begin())>);
  // Doesn't work for SHP. Is it a bug?
  // static_assert(lib::remote_iterator<decltype(dv.segments()[0].begin())>);
  static_assert(lib::distributed_contiguous_range<DV>);
}

TYPED_TEST_P(DistributedVector, Constructors) {
  using DV = typename TypeParam::DV;
  using DVA = typename TypeParam::DVA;
  using V = typename TypeParam::V;

  DV a1(10);
  DVA a2(10);
  TypeParam::iota(a1, 10);
  TypeParam::iota(a2, 10);

  DV a3(10, 2);
  if (comm_rank == 0) {
    V v3(10, 2);
    EXPECT_TRUE(equal(a3, v3));
  }
}

REGISTER_TYPED_TEST_SUITE_P(DistributedVector, Requirements, Constructors);
