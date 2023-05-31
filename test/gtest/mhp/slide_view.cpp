// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Slide : public testing::Test {};

TYPED_TEST_SUITE(Slide, AllTypes);

TYPED_TEST(Slide, is_compliant) {
  TypeParam dv(10, dr::mhp::distribution().halo(2, 1));
  LocalVec<TypeParam> lv(10);
  iota(dv, 100);
  std::iota(rng::begin(lv), rng::end(lv), 100);

  auto local_sliding_view = rng::sliding_view(lv, 4); // halo_bounds + 1
  auto dv_sliding_view [[maybe_unused]] = xhp::views::sliding(dv);

  static_assert(compliant_view<decltype(dv_sliding_view)>);
  EXPECT_TRUE(check_view(local_sliding_view, dv_sliding_view));
}

TYPED_TEST(Slide, segements_are_present) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, dr::mhp::distribution().halo(3));
  const auto dv_segments = dr::ranges::segments(xhp::views::sliding(dv));
  EXPECT_EQ(rng::size(dv_segments), comm_size);
}

TYPED_TEST(Slide, segements_are_present_if_n_equals_halo_plus_1) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, dr::mhp::distribution().halo(3));
  const auto dv_segments = dr::ranges::segments(xhp::views::sliding(dv, 7));
  EXPECT_EQ(rng::size(dv_segments), comm_size);
}

TYPED_TEST(Slide, segements_are_absent_if_n_neq_halo_plus_1) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, dr::mhp::distribution().halo(3));
  EXPECT_DEATH(dr::ranges::segments(xhp::views::sliding(dv, 5)),
               "Assertion .* failed");
}

TYPED_TEST(Slide, can_use_nonlocal_algorithms_with_n_greater_than_halo_plus_1) {
  TypeParam dv(10, dr::mhp::distribution().halo(3));
  iota(dv, 1);
  auto dv_sliding_view = xhp::views::sliding(dv, 8);

  EXPECT_EQ(rng::size(dv_sliding_view), 3);
  EXPECT_TRUE(equal({1, 2, 3, 4, 5, 6, 7, 8}, dv_sliding_view[0]));
  EXPECT_TRUE(equal({2, 3, 4, 5, 6, 7, 8, 9}, dv_sliding_view[1]));
  EXPECT_TRUE(equal({3, 4, 5, 6, 7, 8, 9, 10}, dv_sliding_view[2]));
}

TYPED_TEST(Slide, can_use_nonlocal_algorithms_with_n_less_than_halo_plus_1) {
  TypeParam dv(10, dr::mhp::distribution().halo(3));
  iota(dv, 1);
  auto dv_sliding_view = xhp::views::sliding(dv, 6);

  EXPECT_EQ(rng::size(dv_sliding_view), 5);
  EXPECT_TRUE(equal({1, 2, 3, 4, 5, 6}, dv_sliding_view[0]));
  EXPECT_TRUE(equal({2, 3, 4, 5, 6, 7}, dv_sliding_view[1]));
  // ...
  EXPECT_TRUE(equal({5, 6, 7, 8, 9, 10}, dv_sliding_view[4]));
}

TYPED_TEST(Slide, slide_can_modify_inplace) {
  TypeParam dv(6, dr::mhp::distribution().halo(1));
  iota(dv, 10); // 10,11,12,13,14,15
  dv.halo().exchange();
  xhp::for_each(xhp::views::sliding(dv), [](auto &&r) {
  // SYCL kernel cannot use exceptions
#ifndef SYCL_LANGUAGE_VERSION
    EXPECT_EQ(3, rng::size(r));
#endif
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

TYPED_TEST(Slide, slide_no_halo_works_with_transform) {
  TypeParam dv_in(6);
  TypeParam dv_out(6, 0); // 0,0,0,0,0,0
  iota(dv_in, 10);        // 10,11,12,13,14,15

  xhp::transform(xhp::views::sliding(dv_in), rng::begin(dv_out),
                 [](auto &&v) { return v[0] * 2; });

  EXPECT_EQ(20, dv_out[0]);
  EXPECT_EQ(22, dv_out[1]);
  EXPECT_EQ(24, dv_out[2]);
  // ...
  EXPECT_EQ(30, dv_out[5]);
}

TYPED_TEST(Slide, slide_works_with_transform) {
  TypeParam dv_in(10, dr::mhp::distribution().halo(2));
  // although dv_out having size 6 is sufficient to store result, but its
  // segments have to be aligned with segments of input sliding view, hence size
  // of 10 also in output is required
  TypeParam dv_out(10, 0); // 0,0,0,0,0,0,0,0,0,0
  iota(dv_in, 0);          // 0,1,2,3,4,5,6,7,8,9
  dv_in.halo().exchange();

  xhp::transform(xhp::views::sliding(dv_in), rng::begin(dv_out) + 2,
                 [](auto &&r) { return rng::accumulate(r, 0); });

  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0 + 1 + 2 + 3 + 4, dv_out[2]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6, dv_out[4]);
  EXPECT_EQ(3 + 4 + 5 + 6 + 7, dv_out[5]);
  EXPECT_EQ(4 + 5 + 6 + 7 + 8, dv_out[6]);
  EXPECT_EQ(5 + 6 + 7 + 8 + 9, dv_out[7]);
  EXPECT_EQ(0, dv_out[8]);
  EXPECT_EQ(0, dv_out[9]);
}

// rest of tests is in the Slide3 suite
