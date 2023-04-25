// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "xhp-tests.hpp"

TEST(Take, Basic) {
  dr::shp::distributed_vector<int> x(10);
  x.segments();
  // rng::iota(x, 100);
  fmt::print("x: {}\n", x);
  fmt::print("segments: {}\n", dr::ranges::segments(x));
}
