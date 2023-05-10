// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Slide : public testing::Test {};

TYPED_TEST_SUITE(Slide, AllTypes);

TYPED_TEST(Slide, is_compliant) {
  TypeParam dv(10, dr::halo_bounds(2, 1));
  LocalVec<TypeParam> lv(10);
  iota(dv, 100);
  std::iota(rng::begin(lv), rng::end(lv), 100);

  auto local_sliding_view = rng::sliding_view(lv, 4); // halo_bounds + 1
  auto dv_sliding_view [[maybe_unused]] = xhp::views::sliding(dv, 1, 1);

  static_assert(compliant_view<decltype(dv_sliding_view)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Slide, segements_are_present) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE);
  const auto dv_segments = dr::ranges::segments(dv);
  EXPECT_EQ(rng::size(dv_segments), comm_size);
}

// rest of tests is in the Slide3 suite
