// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;

const std::size_t xdim = 4, ydim = 3, zdim = 2, n2d = xdim * ydim,
                  n3d = xdim * ydim * zdim;
md::extents extents2d(xdim, ydim);
md::extents extents3d(xdim, ydim, zdim);

auto dist2d = dr::mhp::distribution().granularity(ydim);
auto dist3d = dr::mhp::distribution().granularity(ydim * zdim);

TEST(Mdspan, StaticAssert) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);
  static_assert(rng::forward_range<decltype(mdspan)>);
  static_assert(dr::distributed_range<decltype(mdspan)>);
}

TEST(Mdspan, Iterator) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);

  *mdspan.begin() = 17;
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST(Mdspan, Mdindex2D) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  std::size_t i = 1, j = 2;
  dmdspan.mdspan()(i, j) = 17;
  EXPECT_EQ(17, dist[i * ydim + j]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j));
}

TEST(Mdspan, Mdindex3D) {
  xhp::distributed_vector<T> dist(n3d, dist3d);
  auto dmdspan = xhp::views::mdspan(dist, extents3d);

  std::size_t i = 1, j = 2, k = 0;
  dmdspan.mdspan()(i, j, k) = 17;
  EXPECT_EQ(17, dist[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j, k));
}

TEST(Mdspan, Pipe) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto mdspan = dist | xhp::views::mdspan(extents2d);

  *mdspan.begin() = 17;
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST(Mdspan, SegmentIndex2D) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  for (auto segment : dr::ranges::segments(dmdspan)) {
    if (comm_rank == 0 && dr::ranges::rank(segment) == 0) {
      static_assert(std::same_as<T *, decltype(&segment.mdspan()(0, 1))>);
      segment.mdspan()(0, 1) = 99;
      EXPECT_EQ(99, segment[1]);
    }
  }
}

TEST(Mdspan, SegmentExtents) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  // Summing up leading dimension size of segments should equal
  // original
  std::size_t x = 0;
  for (auto segment : dr::ranges::segments(dmdspan)) {
    auto extents = segment.mdspan().extents();
    x += extents.extent(0);
    // Non leading dimension are not changed
    EXPECT_EQ(extents2d.extent(1), extents.extent(1));
  }
  EXPECT_EQ(extents2d.extent(0), x);
}

TEST(Mdspan, Subrange) {
  xhp::distributed_vector<T> dist(n2d, dist2d);
  auto inner = rng::subrange(dist.begin() + ydim, dist.end() - ydim);
  md::extents inner_extents(extents2d.extent(0) - 2, extents2d.extent(1));
  auto dmdspan = xhp::views::mdspan(inner, inner_extents);

  // Summing up leading dimension size of segments should equal
  // original minus 2 rows
  std::size_t x = 0;
  for (auto segment : dr::ranges::segments(dmdspan)) {
    auto extents = segment.mdspan().extents();
    x += extents.extent(0);
    // Non leading dimension are not changed
    EXPECT_EQ(extents2d.extent(1), extents.extent(1));
  }
  EXPECT_EQ(extents2d.extent(0), x + 2);
}
