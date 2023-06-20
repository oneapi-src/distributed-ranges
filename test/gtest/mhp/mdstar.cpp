// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;

const std::size_t rows = 2, cols = 5, n = rows * cols;
md::extents extents(rows, cols);

TEST(Mdspan, StaticAssert) {
  xhp::distributed_vector<T> dist(n);
  auto mdspan = xhp::views::mdspan(dist, extents);
  static_assert(rng::forward_range<decltype(mdspan)>);
  static_assert(dr::distributed_range<decltype(mdspan)>);
}

TEST(Mdspan, Iterator) {
  xhp::distributed_vector<T> dist(n);
  auto mdspan = xhp::views::mdspan(dist, extents);

  *mdspan.begin() = 17;
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST(Mdspan, Mdindex) {
  xhp::distributed_vector<T> dist(n);
  auto dmdspan = xhp::views::mdspan(dist, extents);

  std::size_t i = 1, j = 2;
  dmdspan.mdspan()(i, j) = 17;
  EXPECT_EQ(17, dist[i * cols + j]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j));
}
