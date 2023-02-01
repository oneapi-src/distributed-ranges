// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

struct increment {
  auto operator()(auto &&v) const { v++; }
};

TEST(MhpTests, Subrange) {
  DV dv(10);
  auto r = rng::subrange(dv.begin(), dv.end());
  rng::segments_(r);
  static_assert(lib::distributed_range<decltype(r)>);
}

//
// Zip the segments for 1 or more distributed ranges. e.g.:
//
//   segments(dv1): [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
//   segments(dv2): [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
//
//   drop the first 4 elements and zip the segments for the rest
//
//    zip segments: [[(14, 24)], [(15, 25), (16, 26), (17, 27), (18, 28), (19,
//    29)]]
//
template <typename... Ss> auto zip_segments(Ss &&...iters) {
  auto zip_segment = [](auto &&v) {
    auto zip = [](auto &&...refs) { return rng::views::zip(refs...); };
    return std::apply(zip, v);
  };

  return rng::views::zip(lib::ranges::segments(iters)...) |
         rng::views::transform(zip_segment);
}

//
// Given an iter for a zip, return the segmentation
//
auto zip_iter_segments(auto zip_iter) {
  // Given the list of refs as arguments, convert to list of iters
  auto zip = [](auto &&...refs) { return zip_segments(&refs...); };

  // Convert the zip iterator to a tuple of references, and pass the
  // references as a list of arguments
  return std::apply(zip, *zip_iter);
}

TEST(MhpTests, Zip) {
  DV dv1(10), dv2(10);
  mhp::iota(dv1, 10);
  mhp::iota(dv2, 20);
  auto dzv = rng::view::zip(dv1, dv2);
  fmt::print("dzv: {}\n"
             "  dv1: {}\n"
             "  dv2: {}\n"
             "  segments(dv1): {}\n"
             "  segments(dv2): {}\n",
             dzv, dv1, dv2, lib::ranges::segments(dv1),
             lib::ranges::segments(dv2));
  fmt::print("zip segments: {}\n", zip_iter_segments(dzv.begin() + 4));
}

TEST(MhpTests, Take) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  auto aview = rng::views::take(a, 2);
  auto dv_aview = rng::views::take(dv_a, 2);

  mhp::iota(dv_a, 20);
  if (comm == 0) {
    rng::iota(a, 20);
    EXPECT_TRUE(equal(aview, dv_aview));
  }

  mhp::for_each(dv_aview, increment{});
  if (comm == 0) {
    rng::for_each(aview, increment{});
    EXPECT_TRUE(equal(aview, dv_aview));
  }
}

TEST(MhpTests, Drop) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  auto aview = rng::views::drop(a, 2);
  auto dv_aview = rng::views::drop(dv_a, 2);

  mhp::iota(dv_a, 20);
  if (comm == 0) {
    rng::iota(a, 20);
    EXPECT_TRUE(equal(aview, dv_aview));
  }

  mhp::for_each(dv_aview, increment{});
  if (comm == 0) {
    rng::for_each(aview, increment{});
    EXPECT_TRUE(equal(aview, dv_aview));
  }
}
