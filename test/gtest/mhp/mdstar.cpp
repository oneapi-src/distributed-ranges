// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#if __GNUC__ == 10 && __GNUC_MINOR__ == 4
// mdspan triggers gcc 10 bugs, skip these tests
#else

using T = int;

class Mdspan : public ::testing::Test {
protected:
  std::size_t xdim = 4, ydim = 3, zdim = 2;
  std::size_t n2d = xdim * ydim, n3d = xdim * ydim * zdim;

  std::array<std::size_t, 2> extents2d = {xdim, ydim};
  std::array<std::size_t, 3> extents3d = {xdim, ydim, zdim};

  // 2d data with 1d decompostion
  dr::mhp::distribution dist2d_1d = dr::mhp::distribution().granularity(ydim);
  // 3d data with 1d decompostion
  dr::mhp::distribution dist3d_1d =
      dr::mhp::distribution().granularity(ydim * zdim);
};

using Mdarray = Mdspan;

TEST_F(Mdspan, StaticAssert) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);
  static_assert(rng::forward_range<decltype(mdspan)>);
  static_assert(dr::distributed_range<decltype(mdspan)>);
}

TEST_F(Mdspan, Iterator) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = xhp::views::mdspan(dist, extents2d);

  *mdspan.begin() = 17;
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST_F(Mdspan, Mdindex2D) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  std::size_t i = 1, j = 2;
  dmdspan.mdspan()(i, j) = 17;
  EXPECT_EQ(17, dist[i * ydim + j]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j));
}

TEST_F(Mdspan, Mdindex3D) {
  xhp::distributed_vector<T> dist(n3d, dist3d_1d);
  auto dmdspan = xhp::views::mdspan(dist, extents3d);

  std::size_t i = 1, j = 2, k = 0;
  dmdspan.mdspan()(i, j, k) = 17;
  EXPECT_EQ(17, dist[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j, k));
}

TEST_F(Mdspan, Pipe) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = dist | xhp::views::mdspan(extents2d);

  *mdspan.begin() = 17;
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST_F(Mdspan, SegmentExtents) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);

  // Sum of leading dimension matches original
  std::size_t x = 0;
  for (auto segment : dr::ranges::segments(dmdspan)) {
    auto extents = segment.mdspan().extents();
    x += extents.extent(0);
    // Non leading dimension are not changed
    EXPECT_EQ(extents2d[1], extents.extent(1));
  }
  EXPECT_EQ(extents2d[0], x);
}

TEST_F(Mdspan, Subrange) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto inner = rng::subrange(dist.begin() + ydim, dist.end() - ydim);
  std::array<std::size_t, 2> inner_extents({extents2d[0] - 2, extents2d[1]});
  auto dmdspan = xhp::views::mdspan(inner, inner_extents);

  // Summing up leading dimension size of segments should equal
  // original minus 2 rows
  std::size_t x = 0;
  for (auto segment : dr::ranges::segments(dmdspan)) {
    auto extents = segment.mdspan().extents();
    x += extents.extent(0);
    // Non leading dimension are not changed
    EXPECT_EQ(extents2d[1], extents.extent(1));
  }
  EXPECT_EQ(extents2d[0], x + 2);
}

TEST_F(Mdspan, GridExtents) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  xhp::iota(dist, 100);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);
  auto grid = dmdspan.grid();

  auto x = 0;
  for (std::size_t i = 0; i < grid.extent(0); i++) {
    x += grid(i, 0).mdspan().extent(0);
  }
  EXPECT_EQ(dmdspan.mdspan().extent(0), x);

  auto y = 0;
  for (std::size_t i = 0; i < grid.extent(1); i++) {
    y += grid(0, i).mdspan().extent(1);
  }
  EXPECT_EQ(dmdspan.mdspan().extent(1), y);
}

TEST_F(Mdspan, GridLocalReference) {
  xhp::distributed_vector<T> dist(n2d, dist2d_1d);
  xhp::iota(dist, 100);
  auto dmdspan = xhp::views::mdspan(dist, extents2d);
  auto grid = dmdspan.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mhp::fence();
  EXPECT_EQ(99, dist[0]);
}

TEST_F(Mdarray, StaticAssert) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  static_assert(rng::forward_range<decltype(mdarray)>);
  static_assert(dr::distributed_range<decltype(mdarray)>);
}

TEST_F(Mdarray, Iterator) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);

  *mdarray.begin() = 17;
  EXPECT_EQ(17, *mdarray.begin());
  EXPECT_EQ(17, mdarray[0]);
}

TEST_F(Mdarray, Mdindex2D) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);

  std::size_t i = 1, j = 2;
  mdarray.mdspan()(i, j) = 17;
  EXPECT_EQ(17, mdarray[i * ydim + j]);
  EXPECT_EQ(17, mdarray.mdspan()(i, j));
}

TEST_F(Mdarray, Mdindex3D) {
  xhp::distributed_mdarray<T, 3> mdarray(extents3d);

  std::size_t i = 1, j = 2, k = 0;
  mdarray.mdspan()(i, j, k) = 17;
  EXPECT_EQ(17, mdarray[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, mdarray.mdspan()(i, j, k));
}

TEST_F(Mdarray, GridExtents) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto x = 0;
  for (std::size_t i = 0; i < grid.extent(0); i++) {
    x += grid(i, 0).mdspan().extent(0);
  }
  EXPECT_EQ(mdarray.mdspan().extent(0), x);

  auto y = 0;
  for (std::size_t i = 0; i < grid.extent(1); i++) {
    y += grid(0, i).mdspan().extent(1);
  }
  EXPECT_EQ(mdarray.mdspan().extent(1), y);
}

TEST_F(Mdarray, GridLocalReference) {
  xhp::distributed_mdarray<T, 2> mdarray(extents2d);
  xhp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mhp::fence();
  EXPECT_EQ(99, mdarray[0]);
}

#endif // Skip for gcc 10.4
