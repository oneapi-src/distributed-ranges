// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

TYPED_TEST_P(CommonTests, DistributedVectorRequirements) {
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
  // static_assert(lib::remote_iterator<decltype(dv.segments()[0].begin())>);
  static_assert(lib::distributed_contiguous_range<DV>);
}

TYPED_TEST_P(CommonTests, DistributedVectorConstructors) {
  using DV = typename TypeParam::DV;
  using DVA = typename TypeParam::DVA;

  DV a1(10);
  DVA a2(10);
  TypeParam::iota(a1, 10);
  TypeParam::iota(a2, 10);
}
