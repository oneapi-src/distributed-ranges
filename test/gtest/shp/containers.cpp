// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "containers.hpp"

TYPED_TEST_SUITE(DistributedVectorTest, AllocatorTypes);

TYPED_TEST(DistributedVectorTest, is_random_access_range) {
  static_assert(rng::random_access_range<typename TestFixture::LocalVec>);
  static_assert(rng::random_access_range<const typename TestFixture::LocalVec>);
  static_assert(
      rng::random_access_range<const typename TestFixture::LocalVec &>);
  static_assert(rng::random_access_range<typename TestFixture::DistVec>);
  static_assert(rng::random_access_range<const typename TestFixture::DistVec>);
  static_assert(
      rng::random_access_range<const typename TestFixture::DistVec &>);
}

TYPED_TEST(DistributedVectorTest,
           segments_begins_where_its_creating_iterator_points_to) {
  typename TestFixture::DistVec dv(10);
  std::iota(dv.begin(), dv.end(), 20);

  auto second = dv.begin() + 2;
  EXPECT_EQ(second[0], lib::ranges::segments(second)[0][0]);
}

TYPED_TEST(DistributedVectorTest, fill_constructor) {
  EXPECT_TRUE(equal(typename TestFixture::DistVec(10, 1),
                    typename TestFixture::LocalVec(10, 1)));
}

// TODO: add more tests: fill_constructor_one_item, fill_constructor_no_items

TYPED_TEST(DistributedVectorTest, initializer_list) {
  EXPECT_TRUE(
      equal(typename TestFixture::DistVec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            typename TestFixture::LocalVec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// TODO: add more tests: initializer_list_constructor_one_item,
// initializer_list_constructor_no_items

TYPED_TEST(DistributedVectorTest, Iterator) {
  const int n = 10;
  typename TestFixture::DistVec dv_a(n);
  typename TestFixture::LocalVec v_a(n);

  std::iota(dv_a.begin(), dv_a.end(), 20);
  std::iota(v_a.begin(), v_a.end(), 20);

  EXPECT_TRUE(std::equal(v_a.begin(), v_a.end(), dv_a.begin()));
}
