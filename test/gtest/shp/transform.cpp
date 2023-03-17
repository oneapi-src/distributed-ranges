// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "shp-tests.hpp"

template <typename AllocT> class TransformTest : public testing::Test {
public:
  using DistVec = shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
  constexpr static const auto add_10_func = [](auto x) { return x + 10; };
};

TYPED_TEST_SUITE(TransformTest, AllocatorTypes);

TYPED_TEST(TransformTest, whole_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9};
  shp::transform(shp::par_unseq, a, rng::begin(b), TestFixture::add_10_func);

  EXPECT_TRUE(equal(b, typename TestFixture::LocalVec{10, 11, 12, 13, 14}));
}

TYPED_TEST(TransformTest, whole_non_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {50, 51, 52, 53, 54, 55,
                                     56, 57, 58, 59, 60};

  shp::transform(shp::par_unseq, a, rng::begin(b), TestFixture::add_10_func);

  EXPECT_TRUE(equal(b, typename TestFixture::LocalVec{10, 11, 12, 13, 14, 55,
                                                      56, 57, 58, 59, 60}));
}

TYPED_TEST(TransformTest, part_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9};

  shp::transform(shp::par_unseq, rng::subrange(++rng::begin(a), --rng::end(a)),
                 ++rng::begin(b), TestFixture::add_10_func);

  EXPECT_TRUE(equal(b, typename TestFixture::LocalVec{9, 11, 12, 13, 9}));
}

TYPED_TEST(TransformTest, part_not_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9, 9, 9, 9, 9};

  shp::transform(shp::par_unseq, rng::subrange(++rng::begin(a), rng::end(a)),
                 rng::begin(b) + 5, TestFixture::add_10_func);

  EXPECT_TRUE(
      equal(b, typename TestFixture::LocalVec{9, 9, 9, 9, 9, 11, 12, 13, 9}));
}

TYPED_TEST(TransformTest, large_aligned_whole) {
  const typename TestFixture::DistVec a(12345, 7);
  typename TestFixture::DistVec b(12345, 3);
  shp::transform(shp::par_unseq, a, rng::begin(b), TestFixture::add_10_func);

  EXPECT_EQ(b[0], 17);
  EXPECT_EQ(b[1111], 17);
  EXPECT_EQ(b[2222], 17);
  EXPECT_EQ(b[3333], 17);
  EXPECT_EQ(b[4444], 17);
  EXPECT_EQ(b[5555], 17);
  EXPECT_EQ(b[6666], 17);
  EXPECT_EQ(b[7777], 17);
  EXPECT_EQ(b[8888], 17);
  EXPECT_EQ(b[9999], 17);
  EXPECT_EQ(b[11111], 17);
  EXPECT_EQ(b[12222], 17);
  EXPECT_EQ(b[12344], 17);
}

TYPED_TEST(TransformTest, large_aligned_part) {
  const typename TestFixture::DistVec a(12345, 7);
  typename TestFixture::DistVec b(12345, 3);
  shp::transform(shp::par_unseq,
                 rng::subrange(rng::begin(a) + 1000, rng::begin(a) + 1005),
                 rng::begin(b) + 1000, TestFixture::add_10_func);

  EXPECT_EQ(b[998], 3);
  EXPECT_EQ(b[999], 3);
  EXPECT_EQ(b[1000], 17);
  EXPECT_EQ(b[1001], 17);
  EXPECT_EQ(b[1002], 17);
  EXPECT_EQ(b[1003], 17);
  EXPECT_EQ(b[1004], 17);
  EXPECT_EQ(b[1005], 3);
}

TYPED_TEST(TransformTest, large_aligned_part_shifted) {
  const typename TestFixture::DistVec a(12345, 7);
  typename TestFixture::DistVec b(12345, 3);
  shp::transform(shp::par_unseq,
                 rng::subrange(rng::begin(a) + 1000, rng::begin(a) + 1005),
                 rng::begin(b) + 999, TestFixture::add_10_func);

  EXPECT_EQ(b[998], 3);
  EXPECT_EQ(b[999], 17);
  EXPECT_EQ(b[1000], 17);
  EXPECT_EQ(b[1001], 17);
  EXPECT_EQ(b[1002], 17);
  EXPECT_EQ(b[1003], 17);
  EXPECT_EQ(b[1004], 3);
  EXPECT_EQ(b[1005], 3);
}

TYPED_TEST(TransformTest, large_not_aligned) {
  const typename TestFixture::DistVec a(10000, 7);
  typename TestFixture::DistVec b(17000, 3);
  shp::transform(shp::par_unseq,
                 rng::subrange(rng::begin(a) + 2000, rng::begin(a) + 9000),
                 rng::begin(b) + 9000, TestFixture::add_10_func);

  EXPECT_EQ(b[8999], 3);
  EXPECT_EQ(b[9000], 17);
  EXPECT_EQ(b[9001], 17);

  EXPECT_EQ(b[9999], 17);
  EXPECT_EQ(b[12345], 17);
  EXPECT_EQ(b[13456], 17);
  EXPECT_EQ(b[14567], 17);

  EXPECT_EQ(b[15999], 17);
  EXPECT_EQ(b[16000], 3);
  EXPECT_EQ(b[16001], 3);
}
