// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Slide : public testing::Test {};

TYPED_TEST_SUITE(Slide, AllTypes);

TYPED_TEST(Slide, is_compliant) {
  TypeParam dv(10, dr::halo_bounds(2, 1, false));
  LocalVec<TypeParam> lv(10);
  iota(dv, 100);
  std::iota(rng::begin(lv), rng::end(lv), 100);

  auto local_sliding_view = rng::sliding_view(lv, 4); // halo_bounds + 1
  auto dv_sliding_view [[maybe_unused]] = xhp::views::sliding(dv);

  static_assert(compliant_view<decltype(dv_sliding_view)>);
  EXPECT_TRUE(check_view(local_sliding_view, dv_sliding_view));
}

TYPED_TEST(Slide, segements_are_present) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, dr::halo_bounds(3));
  const auto dv_segments = dr::ranges::segments(xhp::views::sliding(dv));
  EXPECT_EQ(rng::size(dv_segments), comm_size);
}

TYPED_TEST(Slide, slide_can_modify_inplace) {
  TypeParam dv(6, dr::halo_bounds(1));
  iota(dv, 10); // 10,11,12,13,14,15
  dv.halo().exchange();

  xhp::for_each(xhp::views::sliding(dv), [](auto &&r) {
    EXPECT_EQ(3, rng::size(r));
    dr::drlog.debug("assign into {} a sum of {} and {}\n", r[1], r[0], r[2]);
    // watch out that when you use r[0] you get already changed value (or not if
    // halo)
    r[1] += r[2];
  });

  EXPECT_EQ(10, dv[0]);
  EXPECT_EQ(11 + 12, dv[1]);
  EXPECT_EQ(12 + 13, dv[2]);
  EXPECT_EQ(13 + 14, dv[3]);
  EXPECT_EQ(14 + 15, dv[4]);
  EXPECT_EQ(15, dv[5]);
}

// rest of tests is in the Slide3 suite
