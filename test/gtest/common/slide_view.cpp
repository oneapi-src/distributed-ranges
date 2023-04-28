// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mhp/views/sliding.hpp>
#include "xhp-tests.hpp"

template <typename T> class Slide : public testing::Test {
};

TYPED_TEST_SUITE(Slide, AllTypes);

TYPED_TEST(Slide, Basic) {
  Ops1<TypeParam> ops(10);

  auto local = rng::sliding_view(ops.vec, 3);
  rng::begin(local);
  auto dist [[maybe_unused]] = xhp::views::sliding(ops.dist_vec);
  static_assert(dr::ranges::__detail::segments_range<decltype(ranges::segments_(dist))>);

  auto x [[maybe_unused]] = rng::begin(dr::ranges::segments(dist)[0]);
  //static_assert(std::same_as< decltype(x), int>);

  auto y [[maybe_unused]] = dr::ranges::local(x);

  //static_assert(compliant_view<decltype(dist)>);
//  EXPECT_TRUE(check_view(local, dist));
}
