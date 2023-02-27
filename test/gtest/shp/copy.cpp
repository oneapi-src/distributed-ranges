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
  shp::copy_async(rng::begin(dist_vec), rng::end(dist_vec),
                  rng::begin(local_vec))
      .wait();
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_async) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  shp::copy_async(rng::begin(local_vec), rng::end(local_vec),
                  rng::begin(dist_vec))
      .wait();
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_sync) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 9};
  auto ret_it = shp::copy(rng::begin(dist_vec), rng::end(dist_vec),
                          rng::begin(local_vec));
  EXPECT_TRUE(
      equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 9}));
  EXPECT_EQ(*ret_it, 9);
}

TYPED_TEST(CopyTest, local2dist_sync) {
  const typename TestFixture::LocalVec local_vec = {1, 2, 3, 4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 9};
  auto ret_it = shp::copy(rng::begin(local_vec), rng::end(local_vec),
                          rng::begin(dist_vec));
  EXPECT_TRUE(
      equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 9}));
  EXPECT_EQ(*ret_it, 9);
}

TYPED_TEST(CopyTest, dist2local_async_can_interleave) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0};
  auto event_1 =
      shp::copy_async(rng::begin(dist_vec) + 0, rng::begin(dist_vec) + 4,
                      rng::begin(local_vec) + 0);
  auto event_2 =
      shp::copy_async(rng::begin(dist_vec) + 1, rng::begin(dist_vec) + 5,
                      rng::begin(local_vec) + 4);
  event_1.wait();
  event_2.wait();
  EXPECT_TRUE(
      equal(local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, local2dist_async_can_interleave) {
  const typename TestFixture::LocalVec local_vec_1 = {1, 2, 3};
  const typename TestFixture::LocalVec local_vec_2 = {4, 5};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0};
  auto event_1 = shp::copy_async(rng::begin(local_vec_1), rng::end(local_vec_1),
                                 rng::begin(dist_vec));
  auto event_2 = shp::copy_async(rng::begin(local_vec_2), rng::end(local_vec_2),
                                 rng::begin(dist_vec) + 3);
  event_1.wait();
  event_2.wait();
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5}));
}

TYPED_TEST(CopyTest, dist2local_sliced_bothSides) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(rng::begin(dist_vec) + 1, rng::end(dist_vec) - 1,
            rng::begin(local_vec));
  EXPECT_TRUE(equal(
      local_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}

TYPED_TEST(CopyTest, dist2local_sliced_left) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(rng::begin(dist_vec) + 1, rng::end(dist_vec),
            rng::begin(local_vec));
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7,
                                                              8, 9, 10, 0}));
}

TYPED_TEST(CopyTest, dist2local_sliced_right) {
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4, 5,
                                                  6, 7, 8, 9, 10};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(rng::begin(dist_vec), rng::end(dist_vec) - 1,
            rng::begin(local_vec));
  EXPECT_TRUE(equal(
      local_vec, typename TestFixture::LocalVec{1, 2, 3, 4, 5, 6, 7, 8, 9, 0}));
}

TYPED_TEST(CopyTest, local2dist_sliced_bothSides) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(rng::begin(local_vec), rng::end(local_vec),
            rng::begin(dist_vec) + 1);
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{0, 2, 3, 4, 5, 6, 7, 8, 9, 0}));
}

TYPED_TEST(CopyTest, local2dist_sliced_left) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(rng::begin(local_vec), rng::end(local_vec),
            rng::begin(dist_vec) + 2);
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{0, 0, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TYPED_TEST(CopyTest, local2dist_sliced_right) {
  const typename TestFixture::LocalVec local_vec = {2, 3, 4, 5, 6, 7, 8, 9};
  typename TestFixture::DistVec dist_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  shp::copy(rng::begin(local_vec), rng::end(local_vec), rng::begin(dist_vec));
  EXPECT_TRUE(equal(
      dist_vec, typename TestFixture::LocalVec{2, 3, 4, 5, 6, 7, 8, 9, 0, 0}));
}

TYPED_TEST(CopyTest, dev3_dist2local_wholesegment) {
  // when running on 3 devices copy exactly one segment
  const typename TestFixture::DistVec dist_vec = {1, 2, 3, 4,  5,  6,
                                                  7, 8, 9, 10, 11, 12};
  typename TestFixture::LocalVec local_vec = {0, 0, 0, 0};

  auto ret_it = shp::copy(rng::begin(dist_vec) + 4, rng::begin(dist_vec) + 8,
                          rng::begin(local_vec));
  EXPECT_TRUE(equal(local_vec, typename TestFixture::LocalVec{5, 6, 7, 8}));
  EXPECT_EQ(ret_it, rng::end(local_vec));
}

TYPED_TEST(CopyTest, dev3_local2dist_wholesegment) {
  // when running on 3 devices copy into exactly one segment
  const typename TestFixture::LocalVec local_vec = {50, 60, 70, 80};
  typename TestFixture::DistVec dist_vec = {1, 2, 3, 4,  5,  6,
                                            7, 8, 9, 10, 11, 12};
  auto ret_it = shp::copy(rng::begin(local_vec), rng::end(local_vec),
                          rng::begin(dist_vec) + 4);
  EXPECT_TRUE(equal(dist_vec, typename TestFixture::LocalVec{
                                  1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12}));
  EXPECT_EQ(*ret_it, 9);
}
