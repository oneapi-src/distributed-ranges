// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class InclusiveScan : public testing::Test {
public:
};

TYPED_TEST_SUITE(InclusiveScan, AllTypes);

TYPED_TEST(InclusiveScan, whole_range) {
  TypeParam dv_in(6);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(6, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::plus<>());
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
  EXPECT_EQ(1 + 2 + 3, dv_out[2]);
  EXPECT_EQ(1 + 2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5 + 6, dv_out[5]);
}

TYPED_TEST(InclusiveScan, whole_range_with_init_value) {
  TypeParam dv_in(5);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(5, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::plus<>(), 10);
  EXPECT_EQ(10 + 1, dv_out[0]);
  EXPECT_EQ(10 + 1 + 2, dv_out[1]);
  EXPECT_EQ(10 + 1 + 2 + 3, dv_out[2]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(10 + 1 + 2 + 3 + 4 + 5, dv_out[4]);
}

TYPED_TEST(InclusiveScan, empty) {
  TypeParam dv_in(3, 1);
  TypeParam dv_out(3, 0);
  xhp::inclusive_scan(rng::begin(dv_in), rng::begin(dv_in), rng::begin(dv_out));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
}

TYPED_TEST(InclusiveScan, one_element) {
  TypeParam dv_in(3, 1);
  TypeParam dv_out(3, 0);
  xhp::inclusive_scan(rng::begin(dv_in), ++rng::begin(dv_in),
                      rng::begin(dv_out));
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
}

TYPED_TEST(InclusiveScan, multiply) {
  TypeParam dv_in(5);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(5, 0);

  xhp::inclusive_scan(dv_in, dv_out, std::multiplies<>());

  // disabled in sycl because of bug in oneDPL
  // enable back below checks once our PR with bugfix is released
  // see: https://github.com/oneapi-src/oneDPL/pull/1014
#ifndef SYCL_LANGUAGE_VERSION
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 * 2, dv_out[1]);
  EXPECT_EQ(1 * 2 * 3, dv_out[2]);
  EXPECT_EQ(1 * 2 * 3 * 4, dv_out[3]);
  EXPECT_EQ(1 * 2 * 3 * 4 * 5, dv_out[4]);
#endif
}

TYPED_TEST(InclusiveScan, touching_first_segment) {
  TypeParam dv_in(6);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(6, 0);

  xhp::inclusive_scan(rng::begin(dv_in), ++(++rng::begin(dv_in)),
                      rng::begin(dv_out), std::plus<>());
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(0, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
}

TYPED_TEST(InclusiveScan, touching_last_segment) {
  TypeParam dv_in(6);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(6, 0);

  xhp::inclusive_scan(--(--rng::end(dv_in)), rng::end(dv_in),
                      --(--rng::end(dv_out)));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(0, dv_out[1]);
  EXPECT_EQ(0, dv_out[2]);
  EXPECT_EQ(0, dv_out[3]);
  EXPECT_EQ(5, dv_out[4]);
  EXPECT_EQ(5 + 6, dv_out[5]);
}

TYPED_TEST(InclusiveScan, without_last_element) {
  TypeParam dv_in(6);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(6, 0);

  xhp::inclusive_scan(rng::begin(dv_in), --rng::end(dv_in), rng::begin(dv_out));
  EXPECT_EQ(1, dv_out[0]);
  EXPECT_EQ(1 + 2, dv_out[1]);
  EXPECT_EQ(1 + 2 + 3, dv_out[2]);
  EXPECT_EQ(1 + 2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(1 + 2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
}

TYPED_TEST(InclusiveScan, without_first_element) {
  TypeParam dv_in(6);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(6, 0);

  xhp::inclusive_scan(++rng::begin(dv_in), rng::end(dv_in),
                      ++rng::begin(dv_out));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(2, dv_out[1]);
  EXPECT_EQ(2 + 3, dv_out[2]);
  EXPECT_EQ(2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(2 + 3 + 4 + 5 + 6, dv_out[5]);
}

TYPED_TEST(InclusiveScan, without_first_and_last_elements) {
  TypeParam dv_in(6);
  xhp::iota(dv_in, 1);
  TypeParam dv_out(6, 0);

  xhp::inclusive_scan(++rng::begin(dv_in), --rng::end(dv_in),
                      ++rng::begin(dv_out));
  EXPECT_EQ(0, dv_out[0]);
  EXPECT_EQ(2, dv_out[1]);
  EXPECT_EQ(2 + 3, dv_out[2]);
  EXPECT_EQ(2 + 3 + 4, dv_out[3]);
  EXPECT_EQ(2 + 3 + 4 + 5, dv_out[4]);
  EXPECT_EQ(0, dv_out[5]);
}
