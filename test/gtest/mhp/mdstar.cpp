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

#if 0
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
#endif

TEST_F(Mdspan, Grid) {
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

#endif // Skip for gcc 10.4
