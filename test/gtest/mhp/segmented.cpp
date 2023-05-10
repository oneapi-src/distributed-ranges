// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#include <dr/mhp/views/segmented.hpp>

// Fixture
class Segmented : public ::testing::Test {
protected:
  void SetUp() override { dr::mhp::iota(dist_vec, 100); }

  Segmented() : dist_vec(10) {}

  dr::mhp::distributed_vector<int> dist_vec;
};

TEST_F(Segmented, StaticAssert) {
  auto segmented = dr::mhp::segmented_view(rng::views::iota(100),
                                           dr::ranges::segments(dist_vec));
  static_assert(std::forward_iterator<decltype(segmented.begin())>);
  static_assert(rng::forward_range<decltype(segmented)>);
}

TEST_F(Segmented, Basic) {
  auto segmented = dr::mhp::segmented_view(rng::views::iota(100),
                                           dr::ranges::segments(dist_vec));
  EXPECT_EQ(dr::ranges::segments(dist_vec), segmented);
}
