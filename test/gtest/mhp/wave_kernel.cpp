// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

using T = double;
using MDA = dr::mhp::distributed_mdarray<T, 2>;
const int nx = 3, ny = 3;
const std::size_t halo_radius = 1;
auto dist = dr::mhp::distribution().halo(halo_radius);

bool equal(MDA &a, std::vector<T> expected) {
  bool result = true;
  std::size_t k = 0;
  for (std::size_t i = 0; i < a.mdspan().extent(0); i++) {
    for (std::size_t j = 0; j < a.mdspan().extent(1); j++) {
      result = result && a.mdspan()(i, j) == expected[k];
      k++;
    }
  }
  return result;
}

MDA generate_e() {
  MDA a({nx + 1, ny}, dist);
  dr::mhp::fill(a, 0.0);
  for (std::size_t i = 1; i < a.mdspan().extent(0); i++) {
    for (std::size_t j = 0; j < a.mdspan().extent(1); j++) {
      a.mdspan()(i, j) = 10 + i + j;
    }
  }
  dr::mhp::halo(a).exchange();
  return a;
}

MDA generate_u() {
  MDA a({nx + 1, ny}, dist);
  dr::mhp::fill(a, 0.0);
  for (std::size_t i = 0; i < a.mdspan().extent(0); i++) {
    for (std::size_t j = 0; j < a.mdspan().extent(1); j++) {
      a.mdspan()(i, j) = 10 + i + j;
    }
  }
  dr::mhp::halo(a).exchange();
  return a;
}

MDA generate_v() {
  MDA a({nx + 1, ny + 1}, dist);
  dr::mhp::fill(a, 0.0);
  for (std::size_t i = 1; i < a.mdspan().extent(0); i++) {
    for (std::size_t j = 0; j < a.mdspan().extent(1); j++) {
      a.mdspan()(i, j) = 10 + i + j;
    }
  }
  dr::mhp::halo(a).exchange();
  return a;
}

TEST(MhpTests, WaveKernelCreateE) {
  if (comm_size > 1)
    return;
  MDA a = generate_e();
  // first row is unused (all arrays must have the same number of rows)
  // clang-format off
  EXPECT_TRUE(equal(a, { 0,  0,  0,
                        11, 12, 13,
                        12, 13, 14,
                        13, 14, 15}))
      << fmt::format("E:\n{}\n", a);
  // clang-format on
}

TEST(MhpTests, WaveKernelCreateU) {
  if (comm_size > 1)
    return;
  MDA a = generate_u();
  // clang-format off
  EXPECT_TRUE(equal(a, {10, 11, 12,
                        11, 12, 13,
                        12, 13, 14,
                        13, 14, 15}))
      << fmt::format("U:\n{}\n", a);
  // clang-format on
}

TEST(MhpTests, WaveKernelCreateV) {
  if (comm_size > 1)
    return;
  MDA a = generate_v();
  // first row is unused (all arrays must have the same number of rows)
  // clang-format off
  EXPECT_TRUE(
      equal(a, { 0,  0,  0,  0,
                11, 12, 13, 14,
                12, 13, 14, 15,
                13, 14, 15, 16}))
      << fmt::format("V:\n{}\n", a);
  // clang-format on
}

TEST(MhpTests, WaveKernelDeDx) {
  if (comm_size > 1)
    return;
  MDA e = generate_e();
  MDA dedx({nx + 1, ny}, dist);
  dr::mhp::fill(dedx, 0.0);

  auto rhs_dedx = [](auto v) {
    auto [in, out] = v;
    out(0, 0) = in(1, 0) - in(0, 0);
  };
  std::array<std::size_t, 2> start{1, 0};
  std::array<std::size_t, 2> end{
      static_cast<std::size_t>(e.mdspan().extent(0) - 1),
      static_cast<std::size_t>(e.mdspan().extent(1))};
  // FIXME this should work; currently extents require std::size_t type
  // auto e_view = dr::mhp::views::submdspan(e.view(), {1, 0},
  //                                         {e.mdspan().extent(0)-1,
  //                                          e.mdspan().extent(1)});
  auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
  auto dedx_view = dr::mhp::views::submdspan(dedx.view(), start, end);
  dr::mhp::stencil_for_each(rhs_dedx, e_view, dedx_view);

  // first/last rows are boundary data, only interior values are computed
  // clang-format off
  EXPECT_TRUE(equal(dedx, {0, 0, 0,
                           1, 1, 1,
                           1, 1, 1,
                           0, 0, 0}))
      << fmt::format("dedx:\n{}\n", dedx);
  // clang-format on
}

TEST(MhpTests, WaveKernelDeDy) {
  if (comm_size > 1)
    return;
  MDA e = generate_e();
  MDA dedy({nx + 1, ny + 1}, dist);
  dr::mhp::fill(dedy, 0.0);

  auto rhs_dedy = [](auto v) {
    auto [in, out] = v;
    out(0, 0) = in(0, 0) - in(0, -1);
  };

  std::array<std::size_t, 2> start{0, 1};
  std::array<std::size_t, 2> end{
      static_cast<std::size_t>(e.mdspan().extent(0)),
      static_cast<std::size_t>(e.mdspan().extent(1))};
  auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
  auto dedy_view = dr::mhp::views::submdspan(dedy.view(), start, end);
  dr::mhp::stencil_for_each(rhs_dedy, e_view, dedy_view);

  // first row is unused (all arrays must have the same number of rows)
  // first/last cols are boundary data, only interior values are computed
  // clang-format off
  EXPECT_TRUE(equal(dedy, {0, 0, 0, 0,
                           0, 1, 1, 0,
                           0, 1, 1, 0,
                           0, 1, 1, 0}))
      << fmt::format("dedy:\n{}\n", dedy);
  // clang-format on
}

TEST(MhpTests, WaveKernelDuDx) {
  if (comm_size > 1)
    return;
  MDA u = generate_u();
  MDA dudx({nx + 1, ny}, dist);
  dr::mhp::fill(dudx, 0.0);

  auto rhs_dudx = [](auto v) {
    auto [in, out] = v;
    out(0, 0) = in(0, 0) - in(-1, 0);
  };
  std::array<std::size_t, 2> start{1, 0};
  std::array<std::size_t, 2> end{
      static_cast<std::size_t>(u.mdspan().extent(0)),
      static_cast<std::size_t>(u.mdspan().extent(1))};
  auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
  auto dudx_view = dr::mhp::views::submdspan(dudx.view(), start, end);
  dr::mhp::stencil_for_each(rhs_dudx, u_view, dudx_view);

  // first row is unused (all arrays must have the same number of rows)
  // clang-format off
  EXPECT_TRUE(equal(dudx, {0, 0, 0,
                           1, 1, 1,
                           1, 1, 1,
                           1, 1, 1}))
      << fmt::format("dudx:\n{}\n", dudx);
  // clang-format on
}

TEST(MhpTests, WaveKernelDvDy) {
  if (comm_size > 1)
    return;
  MDA v = generate_v();
  MDA dvdy({nx + 1, ny}, dist);
  dr::mhp::fill(dvdy, 0.0);

  auto rhs_dvdy = [](auto v) {
    auto [in, out] = v;
    out(0, 0) = in(0, 1) - in(0, 0);
  };
  std::array<std::size_t, 2> start{1, 0};
  std::array<std::size_t, 2> end{
      static_cast<std::size_t>(v.mdspan().extent(0)),
      static_cast<std::size_t>(v.mdspan().extent(1) - 1)};
  auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
  auto dvdy_view = dr::mhp::views::submdspan(dvdy.view(), start, end);
  dr::mhp::stencil_for_each(rhs_dvdy, v_view, dvdy_view);

  // first row is unused (all arrays must have the same number of rows)
  // clang-format off
  EXPECT_TRUE(equal(dvdy, {0, 0, 0,
                           1, 1, 1,
                           1, 1, 1,
                           1, 1, 1}))
      << fmt::format("dvdy:\n{}\n", dvdy);
  // clang-format on
}

TEST(MhpTests, WaveKernelDivergence) {
  if (comm_size > 1)
    return;
  MDA u = generate_u();
  MDA v = generate_v();
  MDA divuv({nx + 1, ny}, dist);
  dr::mhp::fill(divuv, 0.0);

  auto rhs_div = [](auto args) {
    auto [u, v, out] = args;
    auto dudx = u(0, 0) - u(-1, 0);
    auto dvdy = v(0, 1) - v(0, 0);
    out(0, 0) = dudx + dvdy;
  };
  std::array<std::size_t, 2> start{1, 0};
  std::array<std::size_t, 2> end{
      static_cast<std::size_t>(u.mdspan().extent(0)),
      static_cast<std::size_t>(u.mdspan().extent(1))};
  auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
  auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
  auto divuv_view = dr::mhp::views::submdspan(divuv.view(), start, end);
  dr::mhp::stencil_for_each(rhs_div, u_view, v_view, divuv_view);

  // first row is unused (all arrays must have the same number of rows)
  // clang-format off
  EXPECT_TRUE(equal(divuv, {0, 0, 0,
                            2, 2, 2,
                            2, 2, 2,
                            2, 2, 2}))
      << fmt::format("divuv:\n{}\n", divuv);
  // clang-format on
}
