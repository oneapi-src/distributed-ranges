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

TEST(ShpTests, Copy_async_Dist2Local) {
  const int M = 10;
  const int na = 100, nb = M * na;
  std::size_t n_to_copy = 20;

  V a(n_to_copy), b(M * n_to_copy);
  DV dv_a(na), dv_b(nb);

  std::vector<cl::sycl::event> events;

  std::iota(dv_a.begin(), dv_a.end(), 0);
  std::iota(dv_b.begin(), dv_b.end(), 0);

  for (size_t i = 0, j = 0; i + n_to_copy <= na; i += n_to_copy, j += M * n_to_copy) {
    auto eva = shp::copy_async(dv_a.begin() + i, dv_a.begin() + i + n_to_copy, a.begin());
    events.push_back(eva);

    auto evb = shp::copy_async(dv_b.begin() + j, dv_b.begin() + j + M * n_to_copy, b.begin());
    events.push_back(evb);
    
    sycl::queue q;
    auto root_event = q.submit([=](auto &&h) { h.depends_on(events); });

    auto dv_aview = dv_a | shp::views::slice({i, i + n_to_copy});
    auto dv_bview = dv_b | shp::views::slice({j, j + M * n_to_copy});

    EXPECT_TRUE(equal(a, dv_aview));
    EXPECT_TRUE(equal(b, dv_bview));
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

TEST(ShpTests, Copy_async_Local2Dist) {
  const int M = 10;
  const int na = 100, nb = M * na;
  std::size_t n_to_copy = 20;

  V a(n_to_copy), b(M * n_to_copy);
  DV dv_a(na), dv_b(nb);

  std::vector<cl::sycl::event> events;

  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);

  for (size_t i = 0, j = 0; i + n_to_copy <= na; i += n_to_copy, j += M * n_to_copy) {
    auto eva = shp::copy_async(a.begin(), a.end(), dv_a.begin() + i);
    events.push_back(eva);

    auto evb = shp::copy_async(b.begin(), b.end(), dv_b.begin() + j);
    events.push_back(evb);
    
    sycl::queue q;
    auto root_event = q.submit([=](auto &&h) { h.depends_on(events); });

    auto dv_aview = dv_a | shp::views::slice({i, i + n_to_copy});
    auto dv_bview = dv_b | shp::views::slice({j, j + M * n_to_copy});

    EXPECT_TRUE(equal(a, dv_aview));
    EXPECT_TRUE(equal(b, dv_bview));
  }
}