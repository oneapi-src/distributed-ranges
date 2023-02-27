// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

template <typename AllocT> class CopyTest : public testing::Test {
public:
  using DistVec = shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
};

using AllocatorTypes =
    ::testing::Types<shp::device_allocator<int>,
                     shp::shared_allocator<long long unsigned int>>;
TYPED_TEST_SUITE(CopyTest, AllocatorTypes);

TYPED_TEST(CopyTest, dist2local_async) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0};
  shp::copy_async(dist_vec.begin(), dist_vec.end(), local_vec.begin()).wait();
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_async) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  shp::copy_async(local_vec.begin(), local_vec.end(), dist_vec.begin()).wait();
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_sync) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0};
  shp::copy(dist_vec.begin(), dist_vec.end(), local_vec.begin());
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_sync) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  shp::copy(local_vec.begin(), local_vec.end(), dist_vec.begin());
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_async_can_interleave) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0};
  auto event_1 = shp::copy_async(dist_vec.begin() + 0, dist_vec.begin() + 4,
                                 local_vec.begin() + 0);
  auto event_2 = shp::copy_async(dist_vec.begin() + 1, dist_vec.begin() + 5,
                                 local_vec.begin() + 4);
  event_1.wait();
  event_2.wait();
  EXPECT_TRUE(
      equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_async_can_interleave) {
  const typename TestFixture::LocalVec local_vec_1 = {1, 2, 3};
  const typename TestFixture::LocalVec local_vec_2 = {4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  auto event_1 =
      shp::copy_async(local_vec_1.begin(), local_vec_1.end(), dist_vec.begin());
  auto event_2 = shp::copy_async(local_vec_2.begin(), local_vec_2.end(),
                                 dist_vec.begin() + 3);
  event_1.wait();
  event_2.wait();
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_sliced_bothSides) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(dist_vec.begin() + 1, dist_vec.end() - 1, local_vec.begin());
  EXPECT_TRUE(equal(
      local_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}

TYPED_TEST(CopyTest, dist2local_sliced_left) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(dist_vec.begin() + 1, dist_vec.end(), local_vec.begin());
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7,
                                                              8, 9, 10, 0}));
}

TYPED_TEST(CopyTest, dist2local_sliced_right) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(dist_vec.begin(), dist_vec.end() - 1, local_vec.begin());
  EXPECT_TRUE(equal(
      local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 6, 7, 8, 9, 0}));
}

TYPED_TEST(CopyTest, local2dist_sliced_bothSides) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(local_vec.begin(), local_vec.end(), dist_vec.begin() + 1);
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{0, 2, 3, 4, 5, 6, 7, 8, 9, 0}));
}

TYPED_TEST(CopyTest, local2dist_sliced_left) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(local_vec.begin(), local_vec.end(), dist_vec.begin() + 2);
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{0, 0, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(CopyTest, local2dist_sliced_right) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(local_vec.begin(), local_vec.end(), dist_vec.begin());
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}

TYPED_TEST(CopyTest, dev3_dist2local_wholesegment) {
  // when running on 3 devices copy exactly one segment
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4,  5,  6,
                                                  7, 8, 9, 10, 11, 12};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0};

  shp::copy(dist_vec.begin() + 4, dist_vec.begin() + 8, local_vec.begin());
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{5, 6, 7, 8}));
}

TYPED_TEST(CopyTest, dev3_local2dist_wholesegment) {
  // when running on 3 devices copy into exactly one segment
  const typename TestFixture::LocalVec local_vec = {50, 60, 70, 80};
  typename TestFixture::DistVec dist_vec = {1, 2, 3, 4,  5,  6,
                                            7, 8, 9, 10, 11, 12};
  shp::copy(local_vec.begin(), local_vec.end(), dist_vec.begin() + 4);
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{
                                  1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12}));
}
