// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/detail/mdarray_shim.hpp>

#include "xp-tests.hpp"

#if __GNUC__ == 10 && __GNUC_MINOR__ == 4
// mdspan triggers gcc 10 bugs, skip these tests
#else

using T = int;

// TODO: add tests with ISHMEM backend
class Mdspan : public ::testing::Test {
protected:
  std::size_t xdim = 9, ydim = 5, zdim = 2;
  std::size_t n2d = xdim * ydim, n3d = xdim * ydim * zdim;

  std::array<std::size_t, 2> extents2d = {xdim, ydim};
  std::array<std::size_t, 2> extents2dt = {ydim, xdim};
  std::array<std::size_t, 3> extents3d = {xdim, ydim, zdim};
  std::array<std::size_t, 3> extents3dt = {ydim, zdim, xdim};

  // 2d data with 1d decomposition
  dr::mp::distribution dist2d_1d = dr::mp::distribution().granularity(ydim);
  // 3d data with 1d decomposition
  dr::mp::distribution dist3d_1d =
      dr::mp::distribution().granularity(ydim * zdim);

  std::array<std::size_t, 2> slice_starts = {1, 1};
  std::array<std::size_t, 2> slice_ends = {3, 3};
};

TEST_F(Mdspan, StaticAssert) {
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = xp::views::mdspan(dist, extents2d, dist2d_1d);
  static_assert(rng::forward_range<decltype(mdspan)>);
  static_assert(dr::distributed_range<decltype(mdspan)>);
  auto segments = dr::ranges::segments(mdspan);
  // Begin on a lvalue
  rng::begin(segments);
  // Begin on a rvalue
  // rng::begin(dr::ranges::segments(mdspan));
}

TEST_F(Mdspan, Iterator) {
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = xp::views::mdspan(dist, extents2d, dist2d_1d);

  *mdspan.begin() = 17;
  xp::fence();
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST_F(Mdspan, Mdindex2D) {
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto dmdspan = xp::views::mdspan(dist, extents2d, dist2d_1d);

  std::size_t i = 1, j = 2;
  dmdspan.mdspan()(i, j) = 17;
  xp::fence();
  EXPECT_EQ(17, dist[i * ydim + j]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j));
}

TEST_F(Mdspan, Mdindex3D) {
  xp::distributed_vector<T> dist(n3d, dist3d_1d);
  auto dmdspan = xp::views::mdspan(dist, extents3d, dist3d_1d);

  std::size_t i = 1, j = 2, k = 0;
  dmdspan.mdspan()(i, j, k) = 17;
  xp::fence();
  EXPECT_EQ(17, dist[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, dmdspan.mdspan()(i, j, k));
}

TEST_F(Mdspan, Pipe) {
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto mdspan = dist | xp::views::mdspan(extents2d, dist2d_1d);

  *mdspan.begin() = 17;
  xp::fence();
  EXPECT_EQ(17, *mdspan.begin());
  EXPECT_EQ(17, dist[0]);
}

TEST_F(Mdspan, SegmentExtents) {
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto dmdspan = xp::views::mdspan(dist, extents2d, dist2d_1d);

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
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  auto inner = rng::subrange(dist.begin() + ydim, dist.end() - ydim);
  std::array<std::size_t, 2> inner_extents({extents2d[0] - 2, extents2d[1]});
  auto dmdspan = xp::views::mdspan(inner, inner_extents, dist2d_1d);

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
  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  xp::iota(dist, 100);
  auto dmdspan = xp::views::mdspan(dist, extents2d, dist2d_1d);
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
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xp::distributed_vector<T> dist(n2d, dist2d_1d);
  xp::iota(dist, 100);
  auto dmdspan = xp::views::mdspan(dist, extents2d, dist2d_1d);
  auto grid = dmdspan.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mp::fence();
  EXPECT_EQ(99, dist[0]);
}

using Mdarray = Mdspan;

TEST_F(Mdarray, StaticAssert) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  static_assert(rng::forward_range<decltype(mdarray)>);
  static_assert(dr::distributed_range<decltype(mdarray)>);
  static_assert(dr::distributed_mdspan_range<decltype(mdarray)>);
}

TEST_F(Mdarray, Basic) {
  xp::distributed_mdarray<T, 2> dist(extents2d);
  xp::iota(dist, 100);

  md::mdarray<T, dr::__detail::md_extents<2>> local(xdim, ydim);
  rng::iota(&local(0, 0), &local(0, 0) + local.size(), 100);

  EXPECT_EQ(dist.mdspan(), local);
}

TEST_F(Mdarray, Iterator) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);

  *mdarray.begin() = 17;
  xp::fence();
  EXPECT_EQ(17, *mdarray.begin());
  EXPECT_EQ(17, mdarray[0]);
}

auto mdrange_message(auto &mdarray) {
  return fmt::format("Flat: {}\nMdspan:\n{}", mdarray, mdarray.mdspan());
}

TEST_F(Mdarray, Mdindex2D) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::fill(mdarray, 1);

  std::size_t i = 1, j = 2;
  mdarray.mdspan()(i, j) = 17;
  EXPECT_EQ(17, mdarray[i * ydim + j]);
  EXPECT_EQ(17, mdarray.mdspan()(i, j)) << mdrange_message(mdarray);
}

TEST_F(Mdarray, Mdindex3D) {
  xp::distributed_mdarray<T, 3> mdarray(extents3d);

  std::size_t i = 1, j = 2, k = 0;
  mdarray.mdspan()(i, j, k) = 17;
  xp::fence();
  EXPECT_EQ(17, mdarray[i * ydim * zdim + j * zdim + k]);
  EXPECT_EQ(17, mdarray.mdspan()(i, j, k));
}

TEST_F(Mdarray, GridExtents) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::iota(mdarray, 100);
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
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mp::fence();
  EXPECT_EQ(99, mdarray[0]);
}

TEST_F(Mdarray, Halo) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xp::distributed_mdarray<T, 2> mdarray(extents2d, xp::distribution().halo(1));
  dr::mp::halo(mdarray);
  xp::iota(mdarray, 100);
  auto grid = mdarray.grid();

  auto tile = grid(0, 0).mdspan();
  if (comm_rank == 0) {
    tile(0, 0) = 99;
    EXPECT_EQ(99, tile(0, 0));
  }
  dr::mp::fence();
  EXPECT_EQ(99, mdarray[0]);
}

TEST_F(Mdarray, Enumerate) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  auto e = xp::views::enumerate(mdarray);
  static_assert(dr::distributed_range<decltype(e)>);
}

TEST_F(Mdarray, Slabs) {
  // local_mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  // leading dimension decomp of 3d array creates slabs
  xp::distributed_mdarray<T, 3> mdarray(extents3d);
  for (auto slab : dr::mp::local_mdspans(mdarray)) {
    for (std::size_t i = 0; i < slab.extent(0); i++) {
      for (std::size_t j = 0; j < slab.extent(1); j++) {
        for (std::size_t k = 0; k < slab.extent(2); k++) {
          slab(i, j, k) = 1;
        }
      }
    }
  }
  fence();

  EXPECT_EQ(mdarray.mdspan()(0, 0, 0), 1);
  EXPECT_EQ(
      mdarray.mdspan()(extents3d[0] - 1, extents3d[1] - 1, extents3d[2] - 1),
      1);
}

TEST_F(Mdarray, MdForEach3d) {
  // leading dimension decomp of 3d array creates slabs
  xp::distributed_mdarray<T, 3> mdarray(extents3d);
  std::vector<T> local(extents3d[0] * extents3d[1] * extents3d[2], 0);
  rng::iota(local, 0);

  auto set = [d1 = extents3d[1], d2 = extents3d[2]](auto index, auto v) {
    auto &[o] = v;
    o = index[0] * d1 * d2 + index[1] * d2 + index[2];
  };
  dr::mp::for_each(set, mdarray);

  EXPECT_EQ(xp::views::take(mdarray.view(), local.size()), local)
      << mdrange_message(mdarray);
}

TEST_F(Mdarray, Transpose2D) {
  xp::distributed_mdarray<double, 2> md_in(extents2d), md_out(extents2dt);
  xp::iota(md_in, 100);
  xp::iota(md_out, 200);

  md::mdarray<T, dr::__detail::md_extents<2>> local(extents2dt);
  for (std::size_t i = 0; i < md_out.extent(0); i++) {
    for (std::size_t j = 0; j < md_out.extent(1); j++) {
      local(i, j) = md_in.mdspan()(j, i);
    }
  }

  xp::transpose(md_in, md_out);
  EXPECT_EQ(md_out.mdspan(), local);
}

TEST_F(Mdarray, Transpose3D) {
  xp::distributed_mdarray<double, 3> md_in(extents3d), md_out(extents3dt);
  xp::iota(md_in, 100);
  xp::iota(md_out, 200);

  md::mdarray<T, dr::__detail::md_extents<3>> local(extents3dt);
  for (std::size_t i = 0; i < md_out.extent(0); i++) {
    for (std::size_t j = 0; j < md_out.extent(1); j++) {
      for (std::size_t k = 0; k < md_out.extent(2); k++) {
        local(i, j, k) = md_in.mdspan()(k, i, j);
      }
    }
  }

  xp::transpose(md_in, md_out);
  EXPECT_EQ(local, md_out.mdspan()) << fmt::format("md_in\n{}", md_in.mdspan());
}

using Submdspan = Mdspan;

TEST_F(Submdspan, StaticAssert) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  auto submdspan =
      xp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  static_assert(rng::forward_range<decltype(submdspan)>);
  static_assert(dr::distributed_range<decltype(submdspan)>);
}

TEST_F(Submdspan, Mdindex2D) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::fill(mdarray, 1);
  auto sub = xp::views::submdspan(mdarray.view(), slice_starts, slice_ends);

  std::size_t i = 1, j = 0;
  sub.mdspan()(i, j) = 17;
  xp::fence();
  EXPECT_EQ(17, sub.mdspan()(i, j));
  EXPECT_EQ(17, mdarray.mdspan()(slice_starts[0] + i, slice_starts[1] + j));
  EXPECT_EQ(17, mdarray[(i + slice_starts[0]) * ydim + j + slice_starts[1]])
      << mdrange_message(mdarray);
}

TEST_F(Submdspan, GridExtents) {
  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::iota(mdarray, 100);
  auto sub = xp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  auto grid = sub.grid();
  EXPECT_EQ(slice_ends[0] - slice_starts[0], sub.mdspan().extent(0));
  EXPECT_EQ(slice_ends[1] - slice_starts[1], sub.mdspan().extent(1));

  auto x = 0;
  for (std::size_t i = 0; i < grid.extent(0); i++) {
    x += grid(i, 0).mdspan().extent(0);
  }
  EXPECT_EQ(slice_ends[0] - slice_starts[0], x);
  EXPECT_EQ(slice_ends[0] - slice_starts[0], sub.mdspan().extent(0));

  auto y = 0;
  for (std::size_t i = 0; i < grid.extent(1); i++) {
    y += grid(0, i).mdspan().extent(1);
  }
  EXPECT_EQ(slice_ends[1] - slice_starts[1], y);
  EXPECT_EQ(slice_ends[1] - slice_starts[1], sub.mdspan().extent(1));
}

TEST_F(Submdspan, GridLocalReference) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::iota(mdarray, 100);
  auto sub = xp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  auto grid = sub.grid();

  std::size_t i = 0, j = 0;
  auto tile = grid(0, 0).mdspan();
  if (tile.extent(0) == 0 || tile.extent(1) == 0) {
    return;
  }
  if (comm_rank == 0) {
    tile(i, j) = 99;
    EXPECT_EQ(99, tile(i, j));
  }
  dr::mp::fence();

  auto flat_index = (i + slice_starts[0]) * extents2d[1] + slice_starts[1] + j;
  EXPECT_EQ(99, mdarray[flat_index]) << mdrange_message(mdarray);
}

TEST_F(Submdspan, Segments) {
  // mdspan is not accessible for device memory
  if (options.count("device-memory")) {
    return;
  }

  xp::distributed_mdarray<T, 2> mdarray(extents2d);
  xp::iota(mdarray, 100);
  auto sub = xp::views::submdspan(mdarray.view(), slice_starts, slice_ends);
  auto sspan = sub.mdspan();
  auto sub_segments = dr::ranges::segments(sub);
  using segment_type = rng::range_value_t<decltype(sub_segments)>;

  segment_type first, last;
  bool found_first = false;
  for (auto segment : sub_segments) {
    if (segment.mdspan().extent(0) != 0) {
      if (!found_first) {
        first = segment;
        found_first = true;
      }
      last = segment;
    }
  }

  if (comm_rank == dr::ranges::rank(first)) {
    auto fspan = first.mdspan();
    auto message = fmt::format("Sub:\n{}First:\n{}", sspan, fspan);
    EXPECT_EQ(sspan(0, 0), fspan(0, 0)) << message;
  }

  if (comm_rank == dr::ranges::rank(last)) {
    auto lspan = last.mdspan();
    auto message = fmt::format("Sub:\n{}Last:\n{}", sspan, lspan);
    EXPECT_EQ(sspan(sspan.extent(0) - 1, sspan.extent(1) - 1),
              lspan(lspan.extent(0) - 1, lspan.extent(1) - 1))
        << message;
  }

  dr::mp::barrier();
}

using MdForeach = Mdspan;

TEST_F(MdForeach, 2ops) {
  xp::distributed_mdarray<T, 2> a(extents2d);
  xp::distributed_mdarray<T, 2> b(extents2d);
  auto mda = a.mdspan();
  auto mdb = b.mdspan();
  xp::iota(a, 100);
  xp::iota(b, 200);
  auto copy_op = [](auto v) {
    auto &[in, out] = v;
    out = in;
  };

  xp::for_each(copy_op, a, b);
  EXPECT_EQ(mda(0, 0), mdb(0, 0));
  EXPECT_EQ(mda(xdim - 1, ydim - 1), mdb(xdim - 1, ydim - 1));
}

TEST_F(MdForeach, 3ops) {
  xp::distributed_mdarray<T, 2> a(extents2d);
  xp::distributed_mdarray<T, 2> b(extents2d);
  xp::distributed_mdarray<T, 2> c(extents2d);
  xp::iota(a, 100);
  xp::iota(b, 200);
  xp::iota(c, 200);
  auto copy_op = [](auto v) {
    auto [in1, in2, out] = v;
    out = in1 + in2;
  };

  xp::for_each(copy_op, a, b, c);
  EXPECT_EQ(a.mdspan()(2, 2) + b.mdspan()(2, 2), c.mdspan()(2, 2))
      << fmt::format("A:\n{}\nB:\n{}\nC:\n{}", a.mdspan(), b.mdspan(),
                     c.mdspan());
}

TEST_F(MdForeach, Indexed) {
  xp::distributed_mdarray<T, 2> dist(extents2d);
  auto op = [l = ydim](auto index, auto v) {
    auto &[o] = v;
    o = index[0] * l + index[1];
  };

  xp::for_each(op, dist);
  for (std::size_t i = 0; i < xdim; i++) {
    for (std::size_t j = 0; j < ydim; j++) {
      EXPECT_EQ(dist.mdspan()(i, j), i * ydim + j)
          << fmt::format("i: {} j: {}\n", i, j);
    }
  }
}

using MdStencilForeach = Mdspan;

TEST_F(MdStencilForeach, 2ops) {
  xp::distributed_mdarray<T, 2> a(extents2d);
  xp::distributed_mdarray<T, 2> b(extents2d);
  xp::iota(a, 100);
  xp::iota(b, 200);
  auto mda = a.mdspan();
  auto mdb = b.mdspan();
  auto copy_op = [](auto v) {
    auto [in, out] = v;
    out(0, 0) = in(0, 0);
  };

  xp::stencil_for_each(copy_op, a, b);
  EXPECT_EQ(mda(0, 0), mdb(0, 0));
  EXPECT_EQ(mda(2, 2), mdb(2, 2));
  EXPECT_EQ(mda(xdim - 1, ydim - 1), mdb(xdim - 1, ydim - 1));
}

TEST_F(MdStencilForeach, 3ops) {
  xp::distributed_mdarray<T, 2> a(extents2d);
  xp::distributed_mdarray<T, 2> b(extents2d);
  xp::distributed_mdarray<T, 2> c(extents2d);
  xp::iota(a, 100);
  xp::iota(b, 200);
  xp::iota(c, 300);
  auto copy_op = [](auto v) {
    auto [in1, in2, out] = v;
    out(0, 0) = in1(0, 0) + in2(0, 0);
  };

  xp::stencil_for_each(copy_op, a, b, c);
  EXPECT_EQ(a.mdspan()(2, 2) + b.mdspan()(2, 2), c.mdspan()(2, 2));
}

using MdspanUtil = Mdspan;

template <typename T> struct shadow {
  shadow(std::vector<T> &h) {
    host_ptr = h.data();
    size = h.size();
    device_ptr = alloc.allocate(size);
    copy_in();
  }

  void copy_in() {
    if (dr::mp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      dr::mp::sycl_queue().copy(host_ptr, device_ptr, size).wait();
#else
      assert(false);
#endif
    } else {
      rng::copy(host_ptr, host_ptr + size, device_ptr);
    }
  }

  void flush() {
    if (dr::mp::use_sycl()) {
#ifdef SYCL_LANGUAGE_VERSION
      dr::mp::sycl_queue().copy(device_ptr, host_ptr, size).wait();
#else
      assert(false);
#endif
    } else {
      std::copy(device_ptr, device_ptr + size, host_ptr);
    }
  }

  ~shadow() {
    alloc.deallocate(device_ptr, size);
    device_ptr = nullptr;
  }

  dr::mp::__detail::allocator<T> alloc;
  T *device_ptr = nullptr, *host_ptr = nullptr;
  std::size_t size = 0;
};

TEST_F(MdspanUtil, Pack) {
  std::vector<T> a(n2d);
  std::vector<T> b(n2d);
  rng::iota(a, 100);
  rng::iota(b, 200);

  shadow a_shadow(a);
  shadow b_shadow(b);

  dr::__detail::mdspan_copy(md::mdspan(a_shadow.device_ptr, extents2d),
                            b_shadow.device_ptr)
      .wait();

  a_shadow.flush();
  b_shadow.flush();

  EXPECT_EQ(a, b);
}

TEST_F(MdspanUtil, UnPack) {
  std::vector<T> a(n2d);
  std::vector<T> b(n2d);
  rng::iota(a, 100);
  rng::iota(b, 200);

  shadow a_shadow(a);
  shadow b_shadow(b);

  dr::__detail::mdspan_copy(a_shadow.device_ptr,
                            md::mdspan(b_shadow.device_ptr, extents2d))
      .wait();

  a_shadow.flush();
  b_shadow.flush();

  EXPECT_EQ(a, b);
}

TEST_F(MdspanUtil, Copy) {
  std::vector<T> a(xdim * ydim);
  std::vector<T> b(xdim * ydim);
  rng::iota(a, 100);
  rng::iota(b, 200);

  shadow a_shadow(a);
  shadow b_shadow(b);

  dr::__detail::mdspan_copy(md::mdspan(a_shadow.device_ptr, extents2d),
                            md::mdspan(b_shadow.device_ptr, extents2d))
      .wait();
  a_shadow.flush();
  b_shadow.flush();

  EXPECT_EQ(a, b);
}

TEST_F(MdspanUtil, Transpose2D) {
#ifdef SYCL_LANGUAGE_VERSION
  using vec_alloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
  sycl::queue q;
  vec_alloc my_alloc(q);
  std::vector<T, vec_alloc> a(n2d, my_alloc), b(n2d, my_alloc),
      c(n2d, my_alloc), ref_packed(n2d, my_alloc);
#else
  std::vector<T> a(n2d), b(n2d), c(n2d), ref_packed(n2d);
#endif
  rng::iota(a, 100);
  rng::iota(b, 200);
  rng::iota(c, 300);
  md::mdspan mda(a.data(), extents2d);
  md::mdspan mdc(c.data(), extents2dt);

  md::mdarray<T, dr::__detail::md_extents<2>> ref(extents2dt);
  T *rp = ref_packed.data();
  for (std::size_t i = 0; i < ref.extent(0); i++) {
    for (std::size_t j = 0; j < ref.extent(1); j++) {
      ref(i, j) = mda(j, i);
      *rp++ = ref(i, j);
    }
  }

  // Transpose view
  dr::__detail::mdtranspose<decltype(mda), 1, 0> mdat(mda);
  EXPECT_EQ(ref, mdat);

  // Transpose pack
  dr::__detail::mdspan_copy(mdat, b.begin()).wait();
  EXPECT_EQ(ref_packed, b);

  // Transpose copy
  dr::__detail::mdspan_copy(mdat, mdc).wait();
  EXPECT_EQ(mdat, mdc);
}

TEST_F(MdspanUtil, Transpose3D) {
#ifdef SYCL_LANGUAGE_VERSION
  using vec_alloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
  sycl::queue q;
  vec_alloc my_alloc(q);
  std::vector<T, vec_alloc> ref_packed(n3d, my_alloc), packed(n3d, my_alloc),
      md_data(n3d, my_alloc), mdt_data(n3d, my_alloc);

  md::mdspan<T, dr::__detail::md_extents<3>> md(md_data.data(), extents3d),
      mdt_ref(mdt_data.data(), extents3dt);

#else
  std::vector<T> ref_packed(n3d), packed(n3d), md_data(n3d), mdt_data(n3d);
  md::mdspan<T, dr::__detail::md_extents<3>> md(md_data.data(), extents3d),
      mdt_ref(mdt_data.data(), extents3dt);
#endif

  T *base = &md(0, 0, 0);
  rng::iota(rng::subrange(base, base + md.size()), 100);

  T *rp = ref_packed.data();
  for (std::size_t i = 0; i < mdt_ref.extent(0); i++) {
    for (std::size_t j = 0; j < mdt_ref.extent(1); j++) {
      for (std::size_t k = 0; k < mdt_ref.extent(2); k++) {
        mdt_ref(i, j, k) = md(k, i, j);
        *rp++ = mdt_ref(i, j, k);
      }
    }
  }

  // Transpose view
  dr::__detail::mdtranspose<decltype(md), 2, 0, 1> mdt(md);
  EXPECT_EQ(mdt_ref, mdt) << fmt::format("md:\n{}", md);

  // Transpose pack
  dr::__detail::mdspan_copy(mdt, packed.begin()).wait();
  EXPECT_EQ(ref_packed, packed);
}

#endif // Skip for gcc 10.4
