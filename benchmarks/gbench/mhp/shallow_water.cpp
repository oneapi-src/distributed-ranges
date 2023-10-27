// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include "mpi.h"
#include "wave_utils.hpp"
#include <chrono>
#include <iomanip>

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

namespace ShallowWater {

using T = double;
using Array = dr::mhp::distributed_mdarray<T, 2>;

// gravitational acceleration
constexpr double g = 9.81;
// Coriolis parameter
constexpr double f = 10.0;
// Length scale of geostrophic gyre
constexpr double sigma = 0.4;

void printArray(Array &arr, std::string msg) {
  std::cout << msg << ":\n";
  for (auto segment : dr::ranges::segments(arr)) {
    if (dr::ranges::rank(segment) == std::size_t(comm_rank)) {
      auto origin = segment.origin();
      auto s = segment.mdspan();
      for (std::size_t i = 0; i < s.extent(0); i++) {
        std::size_t global_i = i + origin[0];
        printf("%3zu: ", global_i);
        for (std::size_t j = 0; j < s.extent(1); j++) {
          printf("%9.6f ", s(i, j));
        }
        std::cout << "\n";
      }
      std::cout << "\n" << std::flush;
    }
  }
}

// Get number of read/write bytes and flops for a single time step
// These numbers correspond to the fused kernel version
void calculate_complexity(std::size_t nx, std::size_t ny, std::size_t &nread,
                          std::size_t &nwrite, std::size_t &nflop) {
  // H, dudy, dvdx, q: 2+1+1+3
  // q adv, hv, hu: 1+3+3
  // stage1: 8+8+3 = 19
  // stage2: 9+9+4 = 22
  // stage3: 9+9+4 = 22
  nread = (77 * nx * ny) * sizeof(T);
  // H, dudy, dvdx, q: 1+1+1+1
  // q adv, hv, hu: 4+1+1
  // stage1: 3
  // stage2: 3
  // stage3: 3
  nwrite = (19 * nx * ny) * sizeof(T);
  // H, dudy, dvdx, q: 9+2+2+3
  // q adv, hv, hu: 12+5+5
  // stage1: 36+36+7 = 79
  // stage2: 39+39+10 = 88
  // stage3: 39+39+10 = 88
  nflop = 293 * nx * ny;
}

double exact_elev(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for elevation field.
   *
   * Returns time-dependent elevation of a 2D stationary gyre.
   */
  double amp = 0.02;
  return amp * exp(-(x * x + y * y) / sigma / sigma);
}

double exact_u(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for x velocity field.
   *
   * Returns time-dependent velocity of a 2D stationary gyre.
   */
  double elev = exact_elev(x, y, t, lx, ly);
  return g / f * 2 * y / sigma / sigma * elev;
}

double exact_v(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for y velocity field.
   *
   * Returns time-dependent velocity of a 2D stationary gyre.
   */
  double elev = exact_elev(x, y, t, lx, ly);
  return -g / f * 2 * x / sigma / sigma * elev;
}

double bathymetry(double x, double y, double lx, double ly) {
  /**
   * Bathymetry, i.e. water depth at rest.
   *
   */
  return 1.0;
}

double initial_elev(double x, double y, double lx, double ly) {
  return exact_elev(x, y, 0.0, lx, ly);
}

double initial_u(double x, double y, double lx, double ly) {
  return exact_u(x, y, 0.0, lx, ly);
}

double initial_v(double x, double y, double lx, double ly) {
  return exact_v(x, y, 0.0, lx, ly);
}

// Compute total depth at F points (vertices)
void compute_total_depth(Array &e, Array &h, Array &H_at_f) {
  dr::mhp::halo(e).exchange_finalize();
  // H_at_f: average over 4 adjacent T points, if present
  { // interior part
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.25 * (e(1, 0) + e(1, -1) + e(0, 0) + e(0, -1));
      auto h_f = 0.25 * (h(1, 0) + h(1, -1) + h(0, 0) + h(0, -1));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{e.extent(0) - 1, e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // top row
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(1, 0) + e(1, -1));
      auto h_f = 0.5 * (h(1, 0) + h(1, -1));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{1, e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // bottom row
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(0, 0) + e(0, -1));
      auto h_f = 0.5 * (h(0, 0) + h(0, -1));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{e.extent(0) - 1, 1};
    std::array<std::size_t, 2> end{e.extent(0), e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // left column
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(1, 0) + e(0, 0));
      auto h_f = 0.5 * (h(1, 0) + h(0, 0));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{e.extent(0) - 1, 1};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // right column
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(1, 0) + e(0, 0));
      auto h_f = 0.5 * (h(1, 0) + h(0, 0));
      out(0, 1) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{1, e.extent(1) - 1};
    std::array<std::size_t, 2> end{e.extent(0) - 1, e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // corner (0, 0)
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 0) = e(1, 0) + h(1, 0);
    };
    std::array<std::size_t, 2> start{0, 0};
    std::array<std::size_t, 2> end{1, 1};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // corner (0, end)
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 1) = e(1, 0) + h(1, 0);
    };
    std::array<std::size_t, 2> start{0, e.extent(1) - 1};
    std::array<std::size_t, 2> end{1, e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // corner (end, 0)
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 0) = e(0, 0) + h(0, 0);
    };
    std::array<std::size_t, 2> start{e.extent(0) - 1, 0};
    std::array<std::size_t, 2> end{e.extent(0), 1};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
  { // corner (end, end)
    auto kernel = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 1) = e(0, 0) + h(0, 0);
    };
    std::array<std::size_t, 2> start{e.extent(0) - 1, e.extent(1) - 1};
    std::array<std::size_t, 2> end{e.extent(0), e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, h_view, Hf_view);
  }
}

// Compute auxiliary fields needed for rhs
void compute_aux_fields(Array &u, Array &v, Array &e, Array &hu, Array &hv,
                        Array &dudy, Array &dvdx, Array &H_at_f, Array &q,
                        Array &qa, Array &qb, Array &qg, Array &qd, Array &h,
                        double f, double dx_inv, double dy_inv,
                        bool finalize_halo) {
  { // dudy
    auto kernel = [dy_inv](auto args) {
      auto [u, out] = args;
      out(0, 0) = (u(0, 0) - u(0, -1)) * dy_inv;
    };
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{dudy.extent(0), dudy.extent(1) - 1};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto dudy_view = dr::mhp::views::submdspan(dudy.view(), start, end);
    dr::mhp::stencil_for_each(kernel, u_view, dudy_view);
  }

  compute_total_depth(e, h, H_at_f);

  if (finalize_halo) {
    dr::mhp::halo(v).exchange_finalize();
  }
  { // dvdx
    auto kernel = [dx_inv](auto args) {
      auto [v, out] = args;
      out(0, 0) = (v(1, 0) - v(0, 0)) * dx_inv;
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{dvdx.extent(0) - 1, dvdx.extent(1)};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dvdx_view = dr::mhp::views::submdspan(dvdx.view(), start, end);
    dr::mhp::stencil_for_each(kernel, v_view, dvdx_view);
  }

  { // vorticity
    auto kernel = [f](auto tuple) {
      auto [dudy, dvdx, H_at_f, out] = tuple;
      out(0, 0) = (f - dudy(0, 0) + dvdx(0, 0)) / H_at_f(0, 0);
    };
    dr::mhp::stencil_for_each(kernel, dudy, dvdx, H_at_f, q);
  }
  dr::mhp::halo(q).exchange_begin();
  dr::mhp::halo(q).exchange_finalize();

  { // q advection
    auto kernel = [](auto tuple) {
      auto [q, qa, qb, qg, qd] = tuple;
      auto w = 1. / 12.;
      qa(0, 0) = w * (q(-1, 1) + q(-1, 0) + q(0, 1));
      qb(0, 0) = w * (q(-1, 1) + q(0, 0) + q(0, 1));
      qg(0, 0) = w * (q(-1, 0) + q(0, 0) + q(0, 1));
      qd(0, 0) = w * (q(-1, 0) + q(0, 0) + q(-1, 1));
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{qa.extent(0), qa.extent(1)};
    auto q_view = dr::mhp::views::submdspan(q.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    dr::mhp::stencil_for_each(kernel, q_view, qa_view, qb_view, qg_view,
                              qd_view);
  }
  dr::mhp::halo(qa).exchange_begin();
  dr::mhp::halo(qb).exchange_begin();
  dr::mhp::halo(qg).exchange_begin();
  dr::mhp::halo(qd).exchange_begin();

  { // hv
    auto kernel = [](auto args) {
      auto [v, e, h, out] = args;
      out(0, 0) = 0.5 * (e(0, 0) + h(0, 0) + e(0, -1) + h(0, -1)) * v(0, 0);
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{v.extent(0), v.extent(1) - 1};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    dr::mhp::stencil_for_each(kernel, v_view, e_view, h_view, hv_view);
  }
  dr::mhp::halo(hv).exchange_begin();

  if (finalize_halo) {
    dr::mhp::halo(u).exchange_finalize();
  }
  { // hu
    auto kernel = [](auto args) {
      auto [u, e, h, out] = args;
      out(0, 0) = 0.5 * (e(0, 0) + h(0, 0) + e(1, 0) + h(1, 0)) * u(0, 0);
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{u.extent(0) - 1, u.extent(1)};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    dr::mhp::stencil_for_each(kernel, u_view, e_view, h_view, hu_view);
  }
  dr::mhp::halo(hu).exchange_begin();

  dr::mhp::halo(qa).exchange_finalize();
  dr::mhp::halo(qb).exchange_finalize();
  dr::mhp::halo(qg).exchange_finalize();
  dr::mhp::halo(qd).exchange_finalize();
  dr::mhp::halo(hv).exchange_finalize();
  dr::mhp::halo(hu).exchange_finalize();
}

void rhs(Array &u, Array &v, Array &e, Array &hu, Array &hv, Array &dudy,
         Array &dvdx, Array &H_at_f, Array &q, Array &qa, Array &qb, Array &qg,
         Array &qd, Array &dudt, Array &dvdt, Array &dedt, Array &h, double g,
         double f, double dx_inv, double dy_inv, double dt,
         bool finalize_halo) {
  /**
   * Evaluate right hand side of the equations, vector invariant form
   */
  compute_aux_fields(u, v, e, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, h,
                     f, dx_inv, dy_inv, finalize_halo);
  { // dudt
    auto kernel = [dt, g, dx_inv](auto tuple) {
      auto [e, u, v, hv, qa, qb, qg, qd, out] = tuple;
      // elevation gradient
      auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(0, 0) * u(0, 0) + u(1, 0) * u(1, 0));
      auto v2t_hi = 0.5 * (v(1, 0) * v(1, 0) + v(1, 1) * v(1, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_lo = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedx = (ke_hi - ke_lo) * dx_inv;
      // qhv flux
      auto qhv = qa(1, 0) * hv(1, 1) + qb(0, 0) * hv(0, 1) +
                 qg(0, 0) * hv(0, 0) + qd(1, 0) * hv(1, 0);
      out(0, 0) = dt * (-g * dedx - dkedx + qhv);
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{e.extent(0) - 1, e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto dudt_view = dr::mhp::views::submdspan(dudt.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, u_view, v_view, hv_view, qa_view,
                              qb_view, qg_view, qd_view, dudt_view);
  }

  { // dvdt
    auto kernel = [dt, g, dy_inv](auto tuple) {
      auto [e, u, v, hu, qa, qb, qg, qd, out] = tuple;
      // elevation gradient
      auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_hi = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, -1) * u(-1, -1) + u(0, -1) * u(0, -1));
      auto v2t_lo = 0.5 * (v(0, -1) * v(0, -1) + v(0, 0) * v(0, 0));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedy = (ke_hi - ke_lo) * dy_inv;
      // qhu flux
      auto qhu = qg(0, 0) * hu(0, 0) + qd(0, 0) * hu(-1, 0) +
                 qa(0, -1) * hu(-1, -1) + qb(0, -1) * hu(0, -1);
      out(0, 0) = dt * (-g * dedy - dkedy - qhu);
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{e.extent(0), e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto dvdt_view = dr::mhp::views::submdspan(dvdt.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, u_view, v_view, hu_view, qa_view,
                              qb_view, qg_view, qd_view, dvdt_view);
  }

  { // dedt
    auto kernel = [dt, dx_inv, dy_inv](auto args) {
      auto [hu, hv, out] = args;
      auto dhudx = (hu(0, 0) - hu(-1, 0)) * dx_inv;
      auto dhvdy = (hv(0, 1) - hv(0, 0)) * dy_inv;
      out(0, 0) = -dt * (dhudx + dhvdy);
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{u.extent(0), u.extent(1)};
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto dedt_view = dr::mhp::views::submdspan(dedt.view(), start, end);
    dr::mhp::stencil_for_each(kernel, hu_view, hv_view, dedt_view);
  }
};

void stage1(Array &u, Array &v, Array &e, Array &hu, Array &hv, Array &dudy,
            Array &dvdx, Array &H_at_f, Array &q, Array &qa, Array &qb,
            Array &qg, Array &qd, Array &u1, Array &v1, Array &e1, Array &h,
            double g, double f, double dx_inv, double dy_inv, double dt,
            bool finalize_halo) {
  /**
   * Evaluate stage 1 of the RK time stepper
   *
   * u1 = u + dt*rhs(u)
   *
   */
  compute_aux_fields(u, v, e, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, h,
                     f, dx_inv, dy_inv, finalize_halo);
  { // u update
    auto kernel = [dt, g, dx_inv](auto tuple) {
      auto [e, u, v, hv, qa, qb, qg, qd, out] = tuple;
      // elevation gradient
      auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(0, 0) * u(0, 0) + u(1, 0) * u(1, 0));
      auto v2t_hi = 0.5 * (v(1, 0) * v(1, 0) + v(1, 1) * v(1, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_lo = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedx = (ke_hi - ke_lo) * dx_inv;
      // qhv flux
      auto qhv = qa(1, 0) * hv(1, 1) + qb(0, 0) * hv(0, 1) +
                 qg(0, 0) * hv(0, 0) + qd(1, 0) * hv(1, 0);
      out(0, 0) = u(0, 0) + dt * (-g * dedx - dkedx + qhv);
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{e.extent(0) - 1, e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto u1_view = dr::mhp::views::submdspan(u1.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, u_view, v_view, hv_view, qa_view,
                              qb_view, qg_view, qd_view, u1_view);
  }
  dr::mhp::halo(u1).exchange_begin();

  { // v update
    auto kernel = [dt, g, dy_inv](auto tuple) {
      auto [e, u, v, hu, qa, qb, qg, qd, out] = tuple;
      // elevation gradient
      auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_hi = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, -1) * u(-1, -1) + u(0, -1) * u(0, -1));
      auto v2t_lo = 0.5 * (v(0, -1) * v(0, -1) + v(0, 0) * v(0, 0));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedy = (ke_hi - ke_lo) * dy_inv;
      // qhu flux
      auto qhu = qg(0, 0) * hu(0, 0) + qd(0, 0) * hu(-1, 0) +
                 qa(0, -1) * hu(-1, -1) + qb(0, -1) * hu(0, -1);
      out(0, 0) = v(0, 0) + dt * (-g * dedy - dkedy - qhu);
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{e.extent(0), e.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto v1_view = dr::mhp::views::submdspan(v1.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, u_view, v_view, hu_view, qa_view,
                              qb_view, qg_view, qd_view, v1_view);
  }
  dr::mhp::halo(v1).exchange_begin();

  { // e update
    auto kernel = [dt, dx_inv, dy_inv](auto args) {
      auto [e, hu, hv, out] = args;
      auto dhudx = (hu(0, 0) - hu(-1, 0)) * dx_inv;
      auto dhvdy = (hv(0, 1) - hv(0, 0)) * dy_inv;
      out(0, 0) = e(0, 0) - dt * (dhudx + dhvdy);
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{u.extent(0), u.extent(1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e_view, hu_view, hv_view, e1_view);
  }
  dr::mhp::halo(e1).exchange_begin();
}

void stage2(Array &u, Array &v, Array &e, Array &hu, Array &hv, Array &dudy,
            Array &dvdx, Array &H_at_f, Array &q, Array &qa, Array &qb,
            Array &qg, Array &qd, Array &u1, Array &v1, Array &e1, Array &u2,
            Array &v2, Array &e2, Array &h, double g, double f, double dx_inv,
            double dy_inv, double dt, bool finalize_halo) {
  /**
   * Evaluate stage 2 of the RK time stepper
   *
   * u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
   *
   */
  compute_aux_fields(u1, v1, e1, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd,
                     h, f, dx_inv, dy_inv, finalize_halo);
  { // u update
    auto kernel = [dt, g, dx_inv](auto tuple) {
      auto [e, u, v, hv, qa, qb, qg, qd, u0, out] = tuple;
      // elevation gradient
      auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(0, 0) * u(0, 0) + u(1, 0) * u(1, 0));
      auto v2t_hi = 0.5 * (v(1, 0) * v(1, 0) + v(1, 1) * v(1, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_lo = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedx = (ke_hi - ke_lo) * dx_inv;
      // qhv flux
      auto qhv = qa(1, 0) * hv(1, 1) + qb(0, 0) * hv(0, 1) +
                 qg(0, 0) * hv(0, 0) + qd(1, 0) * hv(1, 0);
      out(0, 0) =
          0.75 * u0(0, 0) + 0.25 * (u(0, 0) + dt * (-g * dedx - dkedx + qhv));
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{e.extent(0) - 1, e.extent(1)};
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    auto u1_view = dr::mhp::views::submdspan(u1.view(), start, end);
    auto v1_view = dr::mhp::views::submdspan(v1.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto u2_view = dr::mhp::views::submdspan(u2.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e1_view, u1_view, v1_view, hv_view,
                              qa_view, qb_view, qg_view, qd_view, u_view,
                              u2_view);
  }
  dr::mhp::halo(u2).exchange_begin();

  { // v update
    auto kernel = [dt, g, dy_inv](auto tuple) {
      auto [e, u, v, hu, qa, qb, qg, qd, v0, out] = tuple;
      // elevation gradient
      auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_hi = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, -1) * u(-1, -1) + u(0, -1) * u(0, -1));
      auto v2t_lo = 0.5 * (v(0, -1) * v(0, -1) + v(0, 0) * v(0, 0));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedy = (ke_hi - ke_lo) * dy_inv;
      // qhu flux
      auto qhu = qg(0, 0) * hu(0, 0) + qd(0, 0) * hu(-1, 0) +
                 qa(0, -1) * hu(-1, -1) + qb(0, -1) * hu(0, -1);
      out(0, 0) =
          0.75 * v0(0, 0) + 0.25 * (v(0, 0) + dt * (-g * dedy - dkedy - qhu));
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{e.extent(0), e.extent(1)};
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    auto u1_view = dr::mhp::views::submdspan(u1.view(), start, end);
    auto v1_view = dr::mhp::views::submdspan(v1.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto v2_view = dr::mhp::views::submdspan(v2.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e1_view, u1_view, v1_view, hu_view,
                              qa_view, qb_view, qg_view, qd_view, v_view,
                              v2_view);
  }
  dr::mhp::halo(v2).exchange_begin();

  { // e update
    auto kernel = [dt, dx_inv, dy_inv](auto args) {
      auto [e, hu, hv, e0, out] = args;
      auto dhudx = (hu(0, 0) - hu(-1, 0)) * dx_inv;
      auto dhvdy = (hv(0, 1) - hv(0, 0)) * dy_inv;
      out(0, 0) = 0.75 * e0(0, 0) + 0.25 * (e(0, 0) - dt * (dhudx + dhvdy));
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{u.extent(0), u.extent(1)};
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e1_view, hu_view, hv_view, e_view,
                              e2_view);
  }
  dr::mhp::halo(e2).exchange_begin();
}

void stage3(Array &u, Array &v, Array &e, Array &hu, Array &hv, Array &dudy,
            Array &dvdx, Array &H_at_f, Array &q, Array &qa, Array &qb,
            Array &qg, Array &qd, Array &u2, Array &v2, Array &e2, Array &h,
            double g, double f, double dx_inv, double dy_inv, double dt,
            bool finalize_halo) {
  /**
   * Evaluate stage 3 of the RK time stepper
   *
   * u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
   *
   */
  compute_aux_fields(u2, v2, e2, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd,
                     h, f, dx_inv, dy_inv, finalize_halo);
  { // u update
    auto kernel = [dt, g, dx_inv](auto tuple) {
      auto [e, u, v, hv, qa, qb, qg, qd, out] = tuple;
      // elevation gradient
      auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(0, 0) * u(0, 0) + u(1, 0) * u(1, 0));
      auto v2t_hi = 0.5 * (v(1, 0) * v(1, 0) + v(1, 1) * v(1, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_lo = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedx = (ke_hi - ke_lo) * dx_inv;
      // qhv flux
      auto qhv = qa(1, 0) * hv(1, 1) + qb(0, 0) * hv(0, 1) +
                 qg(0, 0) * hv(0, 0) + qd(1, 0) * hv(1, 0);
      out(0, 0) *= 1.0 / 3;
      out(0, 0) += 2.0 / 3 * (u(0, 0) + dt * (-g * dedx - dkedx + qhv));
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{e.extent(0) - 1, e.extent(1)};
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    auto u2_view = dr::mhp::views::submdspan(u2.view(), start, end);
    auto v2_view = dr::mhp::views::submdspan(v2.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e2_view, u2_view, v2_view, hv_view,
                              qa_view, qb_view, qg_view, qd_view, u_view);
  }
  dr::mhp::halo(u).exchange_begin();

  { // v update
    auto kernel = [dt, g, dy_inv](auto tuple) {
      auto [e, u, v, hu, qa, qb, qg, qd, out] = tuple;
      // elevation gradient
      auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
      // kinetic energy gradient
      auto u2t_hi = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
      auto v2t_hi = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
      auto ke_hi = 0.5 * (u2t_hi + v2t_hi);
      auto u2t_lo = 0.5 * (u(-1, -1) * u(-1, -1) + u(0, -1) * u(0, -1));
      auto v2t_lo = 0.5 * (v(0, -1) * v(0, -1) + v(0, 0) * v(0, 0));
      auto ke_lo = 0.5 * (u2t_lo + v2t_lo);
      auto dkedy = (ke_hi - ke_lo) * dy_inv;
      // qhu flux
      auto qhu = qg(0, 0) * hu(0, 0) + qd(0, 0) * hu(-1, 0) +
                 qa(0, -1) * hu(-1, -1) + qb(0, -1) * hu(0, -1);
      out(0, 0) *= 1.0 / 3;
      out(0, 0) += 2.0 / 3 * (v(0, 0) + dt * (-g * dedy - dkedy - qhu));
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{e.extent(0), e.extent(1)};
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    auto u2_view = dr::mhp::views::submdspan(u2.view(), start, end);
    auto v2_view = dr::mhp::views::submdspan(v2.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto qa_view = dr::mhp::views::submdspan(qa.view(), start, end);
    auto qb_view = dr::mhp::views::submdspan(qb.view(), start, end);
    auto qg_view = dr::mhp::views::submdspan(qg.view(), start, end);
    auto qd_view = dr::mhp::views::submdspan(qd.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e2_view, u2_view, v2_view, hu_view,
                              qa_view, qb_view, qg_view, qd_view, v_view);
  }
  dr::mhp::halo(v).exchange_begin();

  { // e update
    auto kernel = [dt, dx_inv, dy_inv](auto args) {
      auto [e, hu, hv, out] = args;
      auto dhudx = (hu(0, 0) - hu(-1, 0)) * dx_inv;
      auto dhvdy = (hv(0, 1) - hv(0, 0)) * dy_inv;
      out(0, 0) *= 1.0 / 3;
      out(0, 0) += 2.0 / 3 * (e(0, 0) - dt * (dhudx + dhvdy));
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{u.extent(0), u.extent(1)};
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    dr::mhp::stencil_for_each(kernel, e2_view, hu_view, hv_view, e_view);
  }
  dr::mhp::halo(e).exchange_begin();
}

int run(
    int n, bool benchmark_mode, bool fused_kernels,
    std::function<void()> iter_callback = []() {}) {
  // construct grid
  // number of cells in x, y direction
  std::size_t nx = n;
  std::size_t ny = n;
  const double xmin = -1, xmax = 1;
  const double ymin = -1, ymax = 1;
  ArakawaCGrid grid(xmin, xmax, ymin, ymax, nx, ny);

  std::size_t halo_radius = 1;
  auto dist = dr::mhp::distribution().halo(halo_radius);

  // statistics
  std::size_t nread, nwrite, nflop;
  calculate_complexity(nx, ny, nread, nwrite, nflop);

  if (comm_rank == 0) {
    std::cout << "Using backend: dr" << std::endl;
    if (fused_kernels) {
      std::cout << "Using fused kernels" << std::endl;
    }
    std::cout << "Grid size: " << nx << " x " << ny << std::endl;
    std::cout << "Elevation DOFs: " << nx * ny << std::endl;
    std::cout << "Velocity  DOFs: " << (nx + 1) * ny + nx * (ny + 1)
              << std::endl;
    std::cout << "Total     DOFs: " << nx * ny + (nx + 1) * ny + nx * (ny + 1);
    std::cout << std::endl;
  }

  double t_end = 1.0;
  double t_export = 0.02;

  // state variables
  // water elevation at T points
  Array e({nx + 1, ny}, dist);
  dr::mhp::fill(e, 0.0);
  // x velocity at U points
  Array u({nx + 1, ny}, dist);
  dr::mhp::fill(u, 0.0);
  // y velocity at V points
  Array v({nx + 1, ny + 1}, dist);
  dr::mhp::fill(v, 0.0);

  // bathymetry (water depth at rest) at T points
  Array h({nx + 1, ny}, dist);
  // total depth, e + h, at T points
  Array H({nx + 1, ny}, dist);

  // volume flux
  Array hu({nx + 1, ny}, dist);
  dr::mhp::fill(hu, 0.0);
  Array hv({nx + 1, ny + 1}, dist);
  dr::mhp::fill(hv, 0.0);

  // velocity gradients
  Array dudy({nx + 1, ny + 1}, dist);
  dr::mhp::fill(dudy, 0.0);
  Array dvdx({nx + 1, ny + 1}, dist);
  dr::mhp::fill(dvdx, 0.0);

  // total depth at F points
  Array H_at_f({nx + 1, ny + 1}, dist);
  dr::mhp::fill(H_at_f, 0.0);
  // potential vorticity
  Array q({nx + 1, ny + 1}, dist);
  dr::mhp::fill(q, 0.0);

  // q advection
  Array qa({nx + 1, ny}, dist);
  dr::mhp::fill(qa, 0.0);
  Array qb({nx + 1, ny}, dist);
  dr::mhp::fill(qb, 0.0);
  Array qg({nx + 1, ny}, dist);
  dr::mhp::fill(qg, 0.0);
  Array qd({nx + 1, ny}, dist);
  dr::mhp::fill(qd, 0.0);

  // potential energy
  Array pe({nx + 1, ny}, dist);
  dr::mhp::fill(pe, 0.0);
  // kinetic energy
  Array ke({nx + 1, ny}, dist);
  dr::mhp::fill(ke, 0.0);

  // set bathymetry
  auto h_init_op = [grid, x_offset = grid.dx / 2, y_offset = grid.dy / 2,
                    row_offset = 1](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i >= row_offset) {
      std::size_t global_j = index[1];
      T x = grid.xmin + x_offset + (global_i - row_offset) * grid.dx;
      T y = grid.ymin + y_offset + global_j * grid.dy;
      o = bathymetry(x, y, grid.lx, grid.ly);
    }
  };
  dr::mhp::for_each(h_init_op, h);
  dr::mhp::halo(h).exchange();

  // potential energy offset
  double pe_offset;
  {
    // FIXME easier way of doing this?
    Array h2({nx + 1, ny}, dist);
    auto square = [](auto val) { return val * val; };
    dr::mhp::transform(h, h2.begin(), square);
    auto h2mean = dr::mhp::reduce(h2, static_cast<T>(0), std::plus{}) / nx / ny;
    pe_offset = 0.5 * g * h2mean;
  }

  // state for RK stages
  Array e1({nx + 1, ny}, dist);
  Array u1({nx + 1, ny}, dist);
  Array v1({nx + 1, ny + 1}, dist);
  Array e2({nx + 1, ny}, dist);
  Array u2({nx + 1, ny}, dist);
  Array v2({nx + 1, ny + 1}, dist);

  // time tendencies
  // NOTE not needed if rhs kernels are fused with RK stage assignment
  Array dedt({nx + 1, ny}, dist);
  Array dudt({nx + 1, ny}, dist);
  Array dvdt({nx + 1, ny + 1}, dist);

  // compute time step
  auto max = [](double x, double y) { return std::max(x, y); };
  double h_max = dr::mhp::reduce(h, static_cast<T>(0), max);
  double c = std::sqrt(g * h_max);
  double alpha = 0.5;
  double dt = alpha * std::min(grid.dx, grid.dy) / c;
  dt = t_export / static_cast<int>(ceil(t_export / dt));
  std::size_t nt = static_cast<int>(ceil(t_end / dt));
  if (benchmark_mode) {
    nt = 100;
    dt = 1e-5;
    t_export = 25 * dt;
    t_end = nt * dt;
  }
  if (comm_rank == 0) {
    std::cout << "Time step: " << dt << " s" << std::endl;
    std::cout << "Total run time: " << std::fixed << std::setprecision(1);
    std::cout << t_end << " s, ";
    std::cout << nt << " time steps" << std::endl;
  }

  // set initial conditions
  auto e_init_op = [grid, x_offset = grid.dx / 2, y_offset = grid.dy / 2,
                    row_offset = 1](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i >= row_offset) {
      std::size_t global_j = index[1];
      T x = grid.xmin + x_offset + (global_i - row_offset) * grid.dx;
      T y = grid.ymin + y_offset + global_j * grid.dy;
      o = initial_elev(x, y, grid.lx, grid.ly);
    }
  };
  dr::mhp::for_each(e_init_op, e);
  dr::mhp::halo(e).exchange_begin();

  auto u_init_op = [grid, x_offset = 0.0, y_offset = grid.dy / 2,
                    row_offset = 0](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i >= row_offset) {
      std::size_t global_j = index[1];
      T x = grid.xmin + x_offset + (global_i - row_offset) * grid.dx;
      T y = grid.ymin + y_offset + global_j * grid.dy;
      o = initial_u(x, y, grid.lx, grid.ly);
    }
  };
  dr::mhp::for_each(u_init_op, u);
  dr::mhp::halo(u).exchange_begin();

  auto v_init_op = [grid, x_offset = grid.dx / 2, y_offset = 0.0,
                    row_offset = 1](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i >= row_offset) {
      std::size_t global_j = index[1];
      T x = grid.xmin + x_offset + (global_i - row_offset) * grid.dx;
      T y = grid.ymin + y_offset + global_j * grid.dy;
      o = initial_v(x, y, grid.lx, grid.ly);
    }
  };
  dr::mhp::for_each(v_init_op, v);
  dr::mhp::halo(v).exchange_begin();

  // printArray(h, "Bathymetry");
  // printArray(e, "Initial elev");
  // printArray(u, "Initial u");
  // printArray(v, "Initial v");

  auto add = [](auto ops) { return ops.first + ops.second; };
  auto rk_update2 = [](auto ops) {
    return 0.75 * std::get<0>(ops) +
           0.25 * (std::get<1>(ops) + std::get<2>(ops));
  };
  auto rk_update3 = [](auto ops) {
    return 1.0 / 3.0 * std::get<0>(ops) +
           2.0 / 3.0 * (std::get<1>(ops) + std::get<2>(ops));
  };

  std::size_t i_export = 0;
  double next_t_export = 0.0;
  double t = 0.0;
  double initial_vol = 0.0;
  double initial_ene = 0.0;
  double diff_ene = 0.0;
  bool finalize_halo;
  auto tic = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < nt + 1; i++) {
    t = i * dt;

    if (t >= next_t_export - 1e-8) {
      dr::mhp::halo(u).exchange_finalize();
      dr::mhp::halo(v).exchange_finalize();
      finalize_halo = false;

      double elev_max = dr::mhp::reduce(e, static_cast<T>(0), max);
      double u_max = dr::mhp::reduce(u, static_cast<T>(0), max);
      double q_max = dr::mhp::reduce(q, static_cast<T>(0), max);

      // compute total potential energy
      auto pe_kernel = [](auto args) {
        auto [e, h] = args;
        return 0.5 * g * (e + h) * (e - h);
      };
      dr::mhp::transform(dr::mhp::views::zip(e, h), pe.begin(), pe_kernel);
      double total_pe = ((dr::mhp::reduce(pe, static_cast<T>(0), std::plus{})) +
                         grid.nx * grid.ny * pe_offset) *
                        grid.dx * grid.dy;

      // compute total kinetic energy
      {
        auto kernel = [](auto tuple) {
          auto [e, u, v, h, out] = tuple;
          auto u2_at_t = 0.5 * (u(-1, 0) * u(-1, 0) + u(0, 0) * u(0, 0));
          auto v2_at_t = 0.5 * (v(0, 0) * v(0, 0) + v(0, 1) * v(0, 1));
          auto ke = 0.5 * (u2_at_t + v2_at_t);
          auto H = e(0, 0) + h(0, 0);
          out(0, 0) = H * ke;
        };
        std::array<std::size_t, 2> start{1, 0};
        std::array<std::size_t, 2> end{ke.extent(0), ke.extent(1)};
        auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
        auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
        auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
        auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
        auto ke_view = dr::mhp::views::submdspan(ke.view(), start, end);
        dr::mhp::stencil_for_each(kernel, e_view, u_view, v_view, h_view,
                                  ke_view);
      }
      double total_ke =
          ((dr::mhp::reduce(ke, static_cast<T>(0), std::plus{}))) * grid.dx *
          grid.dy;

      // total energy
      double total_ene = total_pe + total_ke;

      // compute total depth and volume
      dr::mhp::transform(dr::mhp::views::zip(e, h), H.begin(), add);
      double total_vol = (dr::mhp::reduce(H, static_cast<T>(0), std::plus{})) *
                         grid.dx * grid.dy;
      if (i == 0) {
        initial_vol = total_vol;
        initial_ene = total_ene;
      }
      double diff_vol = total_vol - initial_vol;
      diff_ene = total_ene - initial_ene;

      if (comm_rank == 0) {
        printf("%2lu %4lu %.3f ", i_export, i, t);
        printf("elev=%7.8f ", elev_max);
        printf("u=%7.8f ", u_max);
        printf("q=%8.5f ", q_max);
        printf("dV=% 6.3e ", diff_vol);
        printf("PE=%7.2e ", total_pe);
        printf("KE=%7.2e ", total_ke);
        printf("dE=%6.3e", diff_ene);
        printf("\n");
      }
      if (elev_max > 1e3) {
        if (comm_rank == 0) {
          std::cout << "Invalid elevation value: " << elev_max << std::endl;
        }
        return 1;
      }
      i_export += 1;
      next_t_export = i_export * t_export;
    } else {
      finalize_halo = true;
    }

    // step
    iter_callback();
    if (fused_kernels) {
      stage1(u, v, e, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, u1, v1, e1,
             h, g, f, grid.dx_inv, grid.dy_inv, dt, finalize_halo);
      stage2(u, v, e, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, u1, v1, e1,
             u2, v2, e2, h, g, f, grid.dx_inv, grid.dy_inv, dt, true);
      stage3(u, v, e, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, u2, v2, e2,
             h, g, f, grid.dx_inv, grid.dy_inv, dt, true);
    } else {
      // RK stage 1: u1 = u + dt*rhs(u)
      rhs(u, v, e, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, dudt, dvdt,
          dedt, h, g, f, grid.dx_inv, grid.dy_inv, dt, finalize_halo);
      dr::mhp::transform(dr::mhp::views::zip(u, dudt), u1.begin(), add);
      dr::mhp::halo(u1).exchange_begin();
      dr::mhp::transform(dr::mhp::views::zip(v, dvdt), v1.begin(), add);
      dr::mhp::halo(v1).exchange_begin();
      dr::mhp::transform(dr::mhp::views::zip(e, dedt), e1.begin(), add);
      dr::mhp::halo(e1).exchange_begin();

      // RK stage 2: u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
      rhs(u1, v1, e1, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, dudt, dvdt,
          dedt, h, g, f, grid.dx_inv, grid.dy_inv, dt, true);
      dr::mhp::transform(dr::mhp::views::zip(u, u1, dudt), u2.begin(),
                         rk_update2);
      dr::mhp::halo(u2).exchange_begin();
      dr::mhp::transform(dr::mhp::views::zip(v, v1, dvdt), v2.begin(),
                         rk_update2);
      dr::mhp::halo(v2).exchange_begin();
      dr::mhp::transform(dr::mhp::views::zip(e, e1, dedt), e2.begin(),
                         rk_update2);
      dr::mhp::halo(e2).exchange_begin();

      // RK stage 3: u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
      rhs(u2, v2, e2, hu, hv, dudy, dvdx, H_at_f, q, qa, qb, qg, qd, dudt, dvdt,
          dedt, h, g, f, grid.dx_inv, grid.dy_inv, dt, true);
      dr::mhp::transform(dr::mhp::views::zip(u, u2, dudt), u.begin(),
                         rk_update3);
      dr::mhp::halo(u).exchange_begin();
      dr::mhp::transform(dr::mhp::views::zip(v, v2, dvdt), v.begin(),
                         rk_update3);
      dr::mhp::halo(v).exchange_begin();
      dr::mhp::transform(dr::mhp::views::zip(e, e2, dedt), e.begin(),
                         rk_update3);
      dr::mhp::halo(e).exchange_begin();
    }
  }
  dr::mhp::halo(u).exchange_finalize();
  dr::mhp::halo(v).exchange_finalize();
  dr::mhp::halo(e).exchange_finalize();
  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    double t_cpu = duration.count();
    double t_step = t_cpu / nt;
    double read_bw = double(nread) / t_step / (1024 * 1024 * 1024);
    double write_bw = double(nwrite) / t_step / (1024 * 1024 * 1024);
    double flop_rate = double(nflop) / t_step / (1000 * 1000 * 1000);
    double ai = double(nflop) / double(nread + nwrite);
    std::cout << "Duration: " << std::setprecision(3) << t_cpu;
    std::cout << " s" << std::endl;
    std::cout << "Time per step: " << std::setprecision(2) << t_step * 1000;
    std::cout << " ms" << std::endl;
    std::cout << "Reads : " << std::setprecision(3) << read_bw;
    std::cout << " GB/s" << std::endl;
    std::cout << "Writes: " << std::setprecision(3) << write_bw;
    std::cout << " GB/s" << std::endl;
    std::cout << "FLOP/s: " << std::setprecision(3) << flop_rate;
    std::cout << " GFLOP/s" << std::endl;
    std::cout << "Arithmetic intensity: " << std::setprecision(5) << ai;
    std::cout << " FLOP/Byte" << std::endl;
  }

  // printArray(e, "Final elev");
  // printArray(u, "Final u");

  // Compute error against exact solution
  Array e_exact({nx + 1, ny}, dist);
  dr::mhp::fill(e_exact, 0.0);
  Array error({nx + 1, ny}, dist);
  // initial condition for elevation
  auto exact_op = [grid, xmin, ymin, t](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i > 0) {
      std::size_t global_j = index[1];
      T x = xmin + grid.dx / 2 + (global_i - 1) * grid.dx;
      T y = ymin + grid.dy / 2 + global_j * grid.dy;
      o = exact_elev(x, y, t, grid.lx, grid.ly);
    }
  };
  dr::mhp::for_each(exact_op, e_exact);
  dr::mhp::halo(e_exact).exchange();
  auto error_kernel = [](auto ops) {
    auto err = ops.first - ops.second;
    return err * err;
  };
  dr::mhp::transform(dr::mhp::views::zip(e, e_exact), error.begin(),
                     error_kernel);
  double err_L2 = dr::mhp::reduce(error, static_cast<T>(0), std::plus{}) *
                  grid.dx * grid.dy / grid.lx / grid.ly;
  err_L2 = std::sqrt(err_L2);
  if (comm_rank == 0) {
    std::cout << "L2 error: " << std::setw(7) << std::scientific;
    std::cout << std::setprecision(5) << err_L2 << std::endl;
  }

  if (benchmark_mode) {
    return 0;
  }
  if (nx < 128 || ny < 128) {
    if (comm_rank == 0) {
      std::cout << "Skipping correctness test due to small problem size."
                << std::endl;
    }
    return 0;
  }
  if (nx == 128 && ny == 128) {
    double expected_L2 = 4.315799035627906e-05;
    double rel_tolerance = 1e-6;
    double rel_err = err_L2 / expected_L2 - 1.0;
    if (!(fabs(rel_err) < rel_tolerance)) {
      if (comm_rank == 0) {
        std::cout << "ERROR: L2 error deviates from reference value: "
                  << expected_L2 << ", relative error: " << rel_err
                  << std::endl;
      }
      return 1;
    }
  } else {
    double tolerance = 1e-2;
    if (!(err_L2 < tolerance)) {
      if (comm_rank == 0) {
        std::cout << "ERROR: L2 error exceeds tolerance: " << err_L2 << " > "
                  << tolerance << std::endl;
      }
      return 1;
    }
  }
  double ene_tolerance = 1e-8;
  if (!(fabs(diff_ene) < ene_tolerance)) {
    if (comm_rank == 0) {
      std::cout << "ERROR: Energy error exceeds tolerance: |" << diff_ene
                << "| > " << ene_tolerance << std::endl;
    }
    return 1;
  }
  if (comm_rank == 0) {
    std::cout << "SUCCESS" << std::endl;
  }

  return 0;
}

} // namespace ShallowWater

#ifdef STANDALONE_BENCHMARK

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  cxxopts::Options options_spec(argv[0], "wave equation");
  // clang-format off
  options_spec.add_options()
    ("n", "Grid size", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,benchmark-mode", "Run a fixed number of time steps.", cxxopts::value<bool>()->default_value("false"))
    ("sycl", "Execute on SYCL device")
    ("f,fused-kernel", "Use fused kernels.", cxxopts::value<bool>()->default_value("false"))
    ("device-memory", "Use device memory")
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("sycl")) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue q = dr::mhp::select_queue();
    std::cout << "Run on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    dr::mhp::init(q, options.count("device-memory") ? sycl::usm::alloc::device
                                                    : sycl::usm::alloc::shared);
#else
    std::cout << "Sycl support requires icpx\n";
    exit(1);
#endif
  } else {
    if (comm_rank == 0) {
      std::cout << "Run on: CPU\n";
    }
    dr::mhp::init();
  }

  std::size_t n = options["n"].as<std::size_t>();
  bool benchmark_mode = options["t"].as<bool>();
  bool fused_kernels = options["f"].as<bool>();

  auto error = ShallowWater::run(n, benchmark_mode, fused_kernels);
  dr::mhp::finalize();
  MPI_Finalize();
  return error;
}

#else

static void ShallowWater_DR(benchmark::State &state) {

  int n = 1400;
  std::size_t nread, nwrite, nflop;
  ShallowWater::calculate_complexity(n, n, nread, nwrite, nflop);
  Stats stats(state, nread, nwrite, nflop);

  auto iter_callback = [&stats]() { stats.rep(); };
  for (auto _ : state) {
    ShallowWater::run(n, true, true, iter_callback);
  }
}

DR_BENCHMARK(ShallowWater_DR);

#endif
