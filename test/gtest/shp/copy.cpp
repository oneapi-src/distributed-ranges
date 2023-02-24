// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using T = int;
using DV = shp::distributed_vector<T, shp::device_allocator<T>>;
using V = std::vector<T>;

TEST(ShpTests, Copy_async_Dist2Local_simple) {
  const int size = 100;

  DV dv(size);
  V a(size);

  std::iota(dv.begin(), dv.end(), 1);

  shp::copy_async(dv.begin(), dv.end(), a.begin()).wait();

  EXPECT_TRUE(equal(a, dv));
}

TEST(ShpTests, Copy_async_Local2Dist_simple) {
  const int size = 100;

  V a(size);
  DV dv(size);

  std::iota(a.begin(), a.end(), 1);

  shp::copy_async(a.begin(), a.end(), dv.begin()).wait();

  EXPECT_TRUE(equal(a, dv));
}

TEST(ShpTests, Copy_Dist2Local_sliced) {
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

TEST(ShpTests, Copy_async_Dist2Local_sliced) {
  const int M = 10;
  const int na = 100, nb = M * na;
  std::size_t n_to_copy = 20;

  V a(n_to_copy), b(M * n_to_copy);
  DV dv_a(na), dv_b(nb);

  std::iota(dv_a.begin(), dv_a.end(), 0);
  std::iota(dv_b.begin(), dv_b.end(), 0);

  for (size_t i = 0, j = 0; i + n_to_copy <= na;
       i += n_to_copy, j += M * n_to_copy) {
    auto eva = shp::copy_async(dv_a.begin() + i, dv_a.begin() + i + n_to_copy,
                               a.begin());
    auto evb = shp::copy_async(dv_b.begin() + j,
                               dv_b.begin() + j + M * n_to_copy, b.begin());

    eva.wait();
    evb.wait();

    auto dv_aview = dv_a | shp::views::slice({i, i + n_to_copy});
    auto dv_bview = dv_b | shp::views::slice({j, j + M * n_to_copy});

    EXPECT_TRUE(equal(a, dv_aview));
    EXPECT_TRUE(equal(b, dv_bview));
  }
}

TEST(ShpTests, Copy_Local2Dist_sliced) {
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

TEST(ShpTests, Copy_async_Local2Dist_sliced) {
  const int M = 10;
  const int na = 100;
  std::size_t n_to_copy = 20;

  V a(n_to_copy), b(M * n_to_copy);
  DV dv_a(na), dv_b(M * na);

  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 0);

  for (size_t i = 0, j = 0; i + n_to_copy <= na;
       i += n_to_copy, j += M * n_to_copy) {
    auto eva = shp::copy_async(a.begin(), a.end(), dv_a.begin() + i);
    auto evb = shp::copy_async(b.begin(), b.end(), dv_b.begin() + j);

    eva.wait();
    evb.wait();

    auto dv_aview = dv_a | shp::views::slice({i, i + n_to_copy});
    auto dv_bview = dv_b | shp::views::slice({j, j + M * n_to_copy});

    EXPECT_TRUE(equal(a, dv_aview));
    EXPECT_TRUE(equal(b, dv_bview));
  }
}

TEST(ShpTests, Copy_async_Local2Dist_intersegment) {
  const size_t size = 100;
  size_t nproc = shp::nprocs();
  size_t l_size = size / nproc;

  V a(l_size);
  DV dv(size);
  std::vector<cl::sycl::event> events;

  std::iota(a.begin(), a.end(), 1);

  // async operation - synchronisation only after all transfers start
  for (long i = nproc - 2; i >= 0; i--) {
    auto e = shp::copy_async(a.begin(), a.end(),
                             dv.begin() + i * l_size + l_size / 2);
    events.push_back(e);
  }

  auto root_event =
      sycl::queue().submit([=](auto &&h) { h.depends_on(events); });
  root_event.wait();

  for (size_t i = 0; i < nproc - 1; i++) {
    auto dv_view = dv | shp::views::slice({i * l_size + l_size / 2,
                                           (i + 1) * l_size + l_size / 2});
    EXPECT_TRUE(equal(a, dv_view));
  }
}

TEST(ShpTests, Copy_async_Dist2Local_intersegment) {
  const size_t size = 100;
  size_t nproc = shp::nprocs();
  size_t l_size = size / nproc;

  DV dv(size);
  V a[nproc - 1];

  std::vector<cl::sycl::event> events;

  std::iota(dv.begin(), dv.end(), 1);

  // async operation - synchronisation only after all transfers start
  for (long i = nproc - 2; i >= 0; i--) {
    a[i].resize(l_size);
    auto e = shp::copy_async(dv.begin() + i * l_size + l_size / 2,
                             dv.begin() + (i + 1) * l_size + l_size / 2,
                             a[i].begin());
    events.push_back(e);
  }
  sycl::queue q;
  auto root_event = q.submit([=](auto &&h) { h.depends_on(events); });
  root_event.wait();

  for (size_t i = 0; i < nproc - 1; i++) {
    auto dv_view = dv | shp::views::slice({i * l_size + l_size / 2,
                                           (i + 1) * l_size + l_size / 2});
    EXPECT_TRUE(equal(a[i], dv_view));
  }
}

TEST(ShpTests, Copy_async_Local2Dist_midsize) {
  const int size = 2000;

  DV dv1(size), dv2(size), dv3(size);
  V a1(size), a2(size), a3(size);

  std::vector<cl::sycl::event> events;

  std::iota(a1.begin(), a1.end(), 1);
  std::iota(a2.begin(), a2.end(), 1);
  std::iota(a3.begin(), a3.end(), 1);

  // async operation - synchronisation only after all transfers start
  auto ev1 = shp::copy_async(a1.begin(), a1.end(), dv1.begin());
  auto ev2 = shp::copy_async(a2.begin(), a2.end(), dv2.begin());
  auto ev3 = shp::copy_async(a3.begin(), a3.end(), dv3.begin());

  ev1.wait();
  ev2.wait();
  ev3.wait();

  EXPECT_TRUE(equal(a1, dv1));
  EXPECT_TRUE(equal(a2, dv2));
  EXPECT_TRUE(equal(a3, dv3));
}

TEST(ShpTests, Copy_async_Dist2Local_midsize) {
  const int size = 2000;

  DV dv1(size), dv2(size), dv3(size);
  V a1(size), a2(size), a3(size);

  std::iota(dv1.begin(), dv1.end(), 1);
  std::iota(dv2.begin(), dv2.end(), 1);
  std::iota(dv3.begin(), dv3.end(), 1);

  // async operation - synchronisation only after all transfers start
  auto ev1 = shp::copy_async(dv1.begin(), dv1.end(), a1.begin());
  auto ev2 = shp::copy_async(dv2.begin(), dv2.end(), a2.begin());
  auto ev3 = shp::copy_async(dv3.begin(), dv3.end(), a3.begin());

  ev1.wait();
  ev2.wait();
  ev3.wait();

  EXPECT_TRUE(equal(a1, dv1));
  EXPECT_TRUE(equal(a2, dv2));
  EXPECT_TRUE(equal(a3, dv3));
}
