// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using T = int;
using DV = shp::distributed_vector<T, shp::device_allocator<T>>;
using V = std::vector<T>;

TEST(ShpTests, Copy_Dist2Local) {
  const int n = 100;
  std::size_t n_to_copy = 20;

  V a(n_to_copy);
  DV dv_a(n);

  std::iota(dv_a.begin(), dv_a.end(), 0);

  for (size_t i = 0; i + n_to_copy <= n; i += n_to_copy) {
    shp::copy(dv_a.begin() + i, dv_a.begin() + i + n_to_copy, a.begin());

    auto dv_aview = dv_a | shp::views::slice({i, i + n_to_copy});

    EXPECT_TRUE(equal(a, dv_aview));
  }
}

TEST(ShpTests, Copy_Local2Dist) {
  const int n = 100;
  std::size_t n_to_copy = 20;

  V a(n_to_copy);
  DV dv_a(n);

  std::iota(a.begin(), a.end(), 0);

  for (size_t i = 0; i + n_to_copy <= n; i += n_to_copy) {
    shp::copy(a.begin(), a.end(), dv_a.begin() + i);

    auto dv_aview = dv_a | shp::views::slice({i, i + n_to_copy});

    EXPECT_TRUE(equal(a, dv_aview));
  }
}
