// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Slide : public testing::Test {};

TYPED_TEST_SUITE(Slide, AllTypesWithoutIshmem);

TYPED_TEST(Slide, is_compliant) {
  TypeParam dv(10, dr::mhp::distribution().halo(2));
  LocalVec<TypeParam> lv(10);
  iota(dv, 100);
  std::iota(rng::begin(lv), rng::end(lv), 100);

  auto local_sliding_view = rng::sliding_view(lv, 5); // halo_bounds + 1
  auto dv_sliding_view = xhp::views::sliding(dv, 5);

  static_assert(compliant_view<decltype(dv_sliding_view)>);
  EXPECT_TRUE(check_view(local_sliding_view, dv_sliding_view));
}

TYPED_TEST(Slide, is_compliant_even_when_empty) {
  TypeParam dv(10, dr::mhp::distribution().halo(5));
  auto dv_sliding_view = xhp::views::sliding(dv, 11);
  EXPECT_TRUE(rng::empty(dv_sliding_view));
  static_assert(compliant_view<decltype(dv_sliding_view)>);
}

TYPED_TEST(Slide, segements_are_present) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, dr::mhp::distribution().halo(3));
  const auto dv_segments = dr::ranges::segments(xhp::views::sliding(dv, 7));
  EXPECT_EQ(rng::size(dv_segments), comm_size);
}

TYPED_TEST(Slide, segements_are_present_if_n_equals_halo_plus_1) {
  TypeParam dv(EVENLY_DIVIDABLE_SIZE, dr::mhp::distribution().halo(3));
  const auto dv_segments = dr::ranges::segments(xhp::views::sliding(dv, 7));
  EXPECT_EQ(rng::size(dv_segments), comm_size);
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
  xhp::for_each(xhp::views::sliding(dv, 3), [](auto &&r) {
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

  xhp::transform(xhp::views::sliding(dv_in, 1), rng::begin(dv_out),
                 [](auto &&v) { return v[0] * 2; });

  EXPECT_EQ(20, dv_out[0]);
  EXPECT_EQ(22, dv_out[1]);
  EXPECT_EQ(24, dv_out[2]);
  // ...
  EXPECT_EQ(30, dv_out[5]);
}

TYPED_TEST(Slide, slide_works_with_transform_algorithm) {
  TypeParam dv_in(10, dr::mhp::distribution().halo(2));
  // although dv_out having size 6 is sufficient to store result, but its
  // segments have to be aligned with segments of input sliding view, hence size
  // of 10 also in output is required
  TypeParam dv_out(10, 0); // 0,0,0,0,0,0,0,0,0,0
  iota(dv_in, 0);          // 0,1,2,3,4,5,6,7,8,9
  dv_in.halo().exchange();

  xhp::transform(xhp::views::sliding(dv_in, 5), rng::begin(dv_out) + 2,
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

TYPED_TEST(Slide, slide_works_with_transform_view) {
  TypeParam dv(10, dr::mhp::distribution().halo(1));
  iota(dv, 0); // 0,1,2,3,4,5,6,7,8,9
  dv.halo().exchange();

  auto sv = xhp::views::sliding(dv, 3);
  EXPECT_EQ(rng::size(sv), 8);

  auto slided_and_transformed_view = xhp::views::transform(sv, [](auto &&r) {
    // change 3-element range into prefix sum
    // r = r * 2;
    return rng::accumulate(r, 0);
  });

  EXPECT_EQ(rng::size(slided_and_transformed_view), 8);

  xhp::for_each(slided_and_transformed_view, [](auto &&v) {
    // nothing complex can be done without output, just reading value
    typename TypeParam::value_type x [[maybe_unused]] = v;
    x = x + 1;
  });
}

TYPED_TEST(Slide, two_slides_can_be_zipped_and_read_by_foreach) {
  TypeParam dv_1(6, dr::mhp::distribution().halo(1));
  iota(dv_1, 10); // 10,11,12,13,14,15
  dv_1.halo().exchange();

  TypeParam dv_2(6, dr::mhp::distribution().halo(1));
  iota(dv_2, 20); // 20,21,22,23,24,25
  dv_2.halo().exchange();

  xhp::for_each(
      xhp::zip_view(xhp::views::sliding(dv_1, 3), xhp::views::sliding(dv_2, 3)),
      [](auto &&zr) {
        auto &[first, second] = zr;
// SYCL kernel cannot use exceptions
#ifndef SYCL_LANGUAGE_VERSION
        EXPECT_EQ(3, rng::size(first));
        EXPECT_EQ(3, rng::size(second));
#endif
        first[1] = second[0] + second[2];
      });

  dv_1.halo().exchange();
  fence();

  EXPECT_EQ(10, dv_1[0]);
  EXPECT_EQ(20 + 22, dv_1[1]);
  EXPECT_EQ(21 + 23, dv_1[2]);
  EXPECT_EQ(22 + 24, dv_1[3]);
  EXPECT_EQ(23 + 25, dv_1[4]);
  EXPECT_EQ(15, dv_1[5]);
}

TYPED_TEST(Slide, slide_works_on_transformed_range) {
  TypeParam dv(6, dr::mhp::distribution().halo(1));
  iota(dv, 10); // 10,11,12,13,14,15
  dv.halo().exchange();

  auto transformed_dv =
      dv | xhp::views::transform([](auto &&e) { return e * 2; });

  xhp::for_each(xhp::views::sliding(transformed_dv, 3), [](auto &&r) {
  // SYCL kernel cannot use exceptions
#ifndef SYCL_LANGUAGE_VERSION
    EXPECT_EQ(3, rng::size(r));
    // checking that transform indeed happened when we see sliding_view element
    EXPECT_EQ(r[0] % 2, 0);
    EXPECT_EQ(r[1] % 2, 0);
    EXPECT_EQ(r[2] % 2, 0);
    EXPECT_TRUE(r[0] >= 20 && r[0] <= 30);
    EXPECT_TRUE(r[1] >= 20 && r[1] <= 30);
    EXPECT_TRUE(r[2] >= 20 && r[2] <= 30);
#endif
    // nothing sensible without second dv can be done as r is read-only here
    // just test reading these values so icpx -sycl will fail if read is a
    // non-local operation
    typename TypeParam::value_type x [[maybe_unused]] = r[0] + r[2];
    x = x + 1;
  });
}

TEST(ComplexSlide, transform_works_between_vectors_of_arrays) {
  using Row = std::array<int, 6>;
  using DV = xhp::distributed_vector<Row>;

  DV dv_in(6, dr::mhp::distribution().halo(1));
  DV dv_out(6);

  if (dr::mhp::default_comm().rank() == 0) {
    dv_in[0] = Row{11, 12, 13, 14, 15, 16};
    dv_in[1] = Row{21, 22, 23, 24, 25, 26};
    dv_in[2] = Row{31, 32, 33, 34, 35, 36};
    dv_in[3] = Row{41, 42, 43, 44, 45, 46};
    dv_in[4] = Row{51, 52, 53, 54, 55, 56};
    dv_in[5] = Row{61, 62, 63, 64, 65, 66};
  }
  fence();
  dv_in.halo().exchange();

  xhp::for_each(dv_out, [](auto &&row) { rng::fill(row, 0); });

  auto stencil_op = [](auto &&three_rows) {
    auto ret_val = Row{0, 0, 0, 0, 0, 0};
    for (std::size_t i = 1; i < 5; ++i)
      ret_val[i] = three_rows[0][i] + three_rows[1][i - 1] + three_rows[1][i] +
                   three_rows[1][i + 1] + three_rows[2][i];
    return ret_val;
  };

  xhp::transform(xhp::views::sliding(dv_in, 3), rng::begin(dv_out) + 1,
                 stencil_op);

  // computes expected sum of stencil_op when center has given value
  auto c = [](int v) { return v + (v - 10) + (v + 10) + (v - 1) + (v + 1); };

  EXPECT_EQ(static_cast<Row>(dv_out[0]), (Row{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(static_cast<Row>(dv_out[1]),
            (Row{0, c(22), c(23), c(24), c(25), 0}));
  EXPECT_EQ(static_cast<Row>(dv_out[2]),
            (Row{0, c(32), c(33), c(34), c(35), 0}));
  EXPECT_EQ(static_cast<Row>(dv_out[3]),
            (Row{0, c(42), c(43), c(44), c(45), 0}));
  EXPECT_EQ(static_cast<Row>(dv_out[4]),
            (Row{0, c(52), c(53), c(54), c(55), 0}));
  EXPECT_EQ(static_cast<Row>(dv_out[5]), (Row{0, 0, 0, 0, 0, 0}));
}

// rest of tests is in the Slide3 suite
