// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;

const std::size_t xdim = 2, ydim = 3, zdim = 2, n2d = xdim * ydim,
                  n3d = xdim * ydim * zdim;
md::extents extents2d(xdim, ydim);
md::extents extents3d(xdim, ydim, zdim);

TEST(Mdspan, StaticAssert) {
  xhp::distributed_vector<T> dist(n2d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);
  static_assert(rng::forward_range<decltype(mdspan)>);
  static_assert(dr::distributed_range<decltype(mdspan)>);
}

TEST(Mdspan, Iterator) {
  xhp::distributed_vector<T> dist(n2d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);

  *mdspan.begin() = 17;
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST(Mdspan, Mdindex2D) {
  xhp::distributed_vector<T> dist(n2d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  std::size_t i = 1, j = 2;
  dmdspan.mdspan()(i, j) = 17;
  EXPECT_EQ(17, dist[i * ydim + j]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j));
}

TEST(Mdspan, Mdindex3D) {
  xhp::distributed_vector<T> dist(n3d);
  auto dmdspan = xhp::views::mdspan(dist, extents3d);

  std::size_t i = 1, j = 2, k = 0;
  dmdspan.mdspan()(i, j, k) = 17;
  EXPECT_EQ(17, dist[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j, k));
}
