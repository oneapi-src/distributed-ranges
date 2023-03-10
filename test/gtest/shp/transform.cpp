// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "shp-tests.hpp"

namespace drtesting {
std::optional<bool> expect_fast_transform_path;
}

#include "dr/shp/algorithms/transform.hpp"

template <typename AllocT> class TransformTest : public testing::Test {
public:
  using DistVec = shp::distributed_vector<typename AllocT::value_type, AllocT>;
  using LocalVec = std::vector<typename AllocT::value_type>;
  constexpr static const auto add_10_func = std::bind(
      std::plus<typename AllocT::value_type>(), std::placeholders::_1, 10);
};

TYPED_TEST_SUITE(TransformTest, AllocatorTypes);

TYPED_TEST(TransformTest, whole_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9};

  drtesting::expect_fast_transform_path = true;
  shp::transform(shp::par_unseq, a, rng::begin(b), TestFixture::add_10_func);

  EXPECT_TRUE(equal(b, typename TestFixture::LocalVec{10, 11, 12, 13, 14}));
}

TYPED_TEST(TransformTest, whole_non_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {50, 51, 52, 53, 54, 55,
                                     56, 57, 58, 59, 60};

  drtesting::expect_fast_transform_path = false;
  shp::transform(shp::par_unseq, a, rng::begin(b), TestFixture::add_10_func);

  EXPECT_TRUE(equal(b, typename TestFixture::LocalVec{10, 11, 12, 13, 14, 55,
                                                      56, 57, 58, 59, 60}));
}

TYPED_TEST(TransformTest, part_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3, 4};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9};

  drtesting::expect_fast_transform_path = true;
  shp::transform(shp::par_unseq, rng::subrange(++rng::begin(a), --rng::end(a)),
                 ++rng::begin(b), TestFixture::add_10_func);

  EXPECT_TRUE(equal(b, typename TestFixture::LocalVec{9, 11, 12, 13, 9}));
}

TYPED_TEST(TransformTest, part_not_aligned) {
  const typename TestFixture::DistVec a = {0, 1, 2, 3};
  typename TestFixture::DistVec b = {9, 9, 9, 9, 9, 9, 9, 9, 9};

  drtesting::expect_fast_transform_path = false;
  shp::transform(shp::par_unseq, rng::subrange(++rng::begin(a), rng::end(a)),
                 rng::begin(b) + 5, TestFixture::add_10_func);

  EXPECT_TRUE(
      equal(b, typename TestFixture::LocalVec{9, 9, 9, 9, 9, 11, 12, 13, 9}));
}
