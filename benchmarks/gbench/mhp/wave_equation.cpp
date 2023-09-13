// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include "mpi.h"
#include <chrono>
#include <iomanip>

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

using T = double;

using Array = dr::mhp::distributed_mdarray<T, 2>;

// gravitational acceleration
constexpr double g = 9.81;
// water depth
constexpr double h = 1.0;

// Get number of read/write bytes and flops for a single time step
// These numbers correspond to the fused kernel version
void calculate_complexity(std::size_t nx, std::size_t ny, std::size_t &nread,
                          std::size_t &nwrite, std::size_t &nflop) {
  // stage1: 2+2+3 = 7
  // stage2: 3+3+4 = 10
  // stage3: 3+3+4 = 10
  nread = (27 * nx * ny) * sizeof(T);
  // stage1: 3
  // stage2: 3
  // stage3: 3
  nwrite = (9 * nx * ny) * sizeof(T);
  // stage1: 3+3+6 = 12
  // stage2: 6+6+9 = 21
  // stage3: 6+6+9 = 21
  nflop = 54 * nx * ny;
}

double exact_elev(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for elevation field.
   *
   * Returns time-dependent elevation of a 2D standing wave in a
   * rectangular domain.
   */
  double amp = 0.5;
  double c = std::sqrt(g * h);
  std::size_t n = 1;
  double sol_x = std::cos(2 * n * M_PI * x / lx);
  std::size_t m = 1;
  double sol_y = std::cos(2 * m * M_PI * y / ly);
  double omega = c * M_PI * std::hypot(n / lx, m / ly);
  double sol_t = std::cos(2 * omega * t);
  return amp * sol_x * sol_y * sol_t;
}

double initial_elev(double x, double y, double lx, double ly) {
  return exact_elev(x, y, 0.0, lx, ly);
}

void rhs(Array &u, Array &v, Array &e, Array &dudt, Array &dvdt, Array &dedt,
         double g, double h, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate right hand side of the equations
   */

  auto rhs_dedx = [dt, g, dx_inv](auto v) {
    auto [in, out] = v;
    out(0, 0) = -dt * g * (in(1, 0) - in(0, 0)) * dx_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0) - 1),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto dudt_view = dr::mhp::views::submdspan(dudt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dedx, e_view, dudt_view);
  }

  auto rhs_dedy = [dt, g, dy_inv](auto v) {
    auto [in, out] = v;
    out(0, 0) = -dt * g * (in(0, 0) - in(0, -1)) * dy_inv;
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0)),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto dvdt_view = dr::mhp::views::submdspan(dvdt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dedy, e_view, dvdt_view);
  }

  auto rhs_div = [dt, h, dx_inv, dy_inv](auto args) {
    auto [u, v, out] = args;
    auto dudx = (u(0, 0) - u(-1, 0)) * dx_inv;
    auto dvdy = (v(0, 1) - v(0, 0)) * dy_inv;
    out(0, 0) = -dt * h * (dudx + dvdy);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(u.mdspan().extent(0)),
        static_cast<std::size_t>(u.mdspan().extent(1))};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dedt_view = dr::mhp::views::submdspan(dedt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_div, u_view, v_view, dedt_view);
  }
};

void stage1(Array &u, Array &v, Array &e, Array &u1, Array &v1, Array &e1,
            double g, double h, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate stage 1 of the RK time stepper
   *
   * u1 = u + dt*rhs(u)
   *
   */
  // u: elevation x gradient
  auto rhs_u1 = [dt, g, dx_inv](auto tuple) {
    auto [e, u, out] = tuple;
    auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
    out(0, 0) = u(0, 0) - dt * g * dedx;
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0) - 1),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto u1_view = dr::mhp::views::submdspan(u1.view(), start, end);
    dr::mhp::stencil_for_each(rhs_u1, e_view, u_view, u1_view);
    dr::mhp::halo(u1).exchange();
  }

  // v: elevation y gradient
  auto rhs_v1 = [dt, g, dy_inv](auto tuple) {
    auto [e, v, out] = tuple;
    auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
    out(0, 0) = v(0, 0) - dt * g * dedy;
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0)),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto v1_view = dr::mhp::views::submdspan(v1.view(), start, end);
    dr::mhp::stencil_for_each(rhs_v1, e_view, v_view, v1_view);
    dr::mhp::halo(v1).exchange();
  }

  // e: divergence of (u, v)
  auto rhs_e1 = [dt, h, dx_inv, dy_inv](auto tuple) {
    auto [e, u, v, out] = tuple;
    auto dudx = (u(0, 0) - u(-1, 0)) * dx_inv;
    auto dvdy = (v(0, 1) - v(0, 0)) * dy_inv;
    out(0, 0) = e(0, 0) - dt * h * (dudx + dvdy);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(u.mdspan().extent(0)),
        static_cast<std::size_t>(u.mdspan().extent(1))};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    dr::mhp::stencil_for_each(rhs_e1, e_view, u_view, v_view, e1_view);
    dr::mhp::halo(e1).exchange();
  }
};

void stage2(Array &u, Array &v, Array &e, Array &u1, Array &v1, Array &e1,
            Array &u2, Array &v2, Array &e2, double g, double h, double dx_inv,
            double dy_inv, double dt) {
  /**
   * Evaluate stage 2 of the RK time stepper
   *
   * u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
   *
   */
  // u: elevation x gradient
  auto rhs_u2 = [dt, g, dx_inv](auto tuple) {
    auto [e1, u1, u, out] = tuple;
    auto dedx = (e1(1, 0) - e1(0, 0)) * dx_inv;
    out(0, 0) = 0.75 * u(0, 0) + 0.25 * (u1(0, 0) - dt * g * dedx);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0) - 1),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    auto u1_view = dr::mhp::views::submdspan(u1.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto u2_view = dr::mhp::views::submdspan(u2.view(), start, end);
    dr::mhp::stencil_for_each(rhs_u2, e1_view, u1_view, u_view, u2_view);
    dr::mhp::halo(u2).exchange();
  }

  // v: elevation y gradient
  auto rhs_v2 = [dt, g, dy_inv](auto tuple) {
    auto [e1, v1, v, out] = tuple;
    auto dedy = (e1(0, 0) - e1(0, -1)) * dy_inv;
    out(0, 0) = 0.75 * v(0, 0) + 0.25 * (v1(0, 0) - dt * g * dedy);
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0)),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    auto v1_view = dr::mhp::views::submdspan(v1.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto v2_view = dr::mhp::views::submdspan(v2.view(), start, end);
    dr::mhp::stencil_for_each(rhs_v2, e1_view, v1_view, v_view, v2_view);
    dr::mhp::halo(v2).exchange();
  }

  // e: divergence of (u, v)
  auto rhs_e2 = [dt, h, dx_inv, dy_inv](auto tuple) {
    auto [e1, u1, v1, e, out] = tuple;
    auto dudx = (u1(0, 0) - u1(-1, 0)) * dx_inv;
    auto dvdy = (v1(0, 1) - v1(0, 0)) * dy_inv;
    out(0, 0) = 0.75 * e(0, 0) + 0.25 * (e1(0, 0) - dt * h * (dudx + dvdy));
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(u.mdspan().extent(0)),
        static_cast<std::size_t>(u.mdspan().extent(1))};
    auto e1_view = dr::mhp::views::submdspan(e1.view(), start, end);
    auto u1_view = dr::mhp::views::submdspan(u1.view(), start, end);
    auto v1_view = dr::mhp::views::submdspan(v1.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    dr::mhp::stencil_for_each(rhs_e2, e1_view, u1_view, v1_view, e_view,
                              e2_view);
    dr::mhp::halo(e2).exchange();
  }
};

void stage3(Array &u, Array &v, Array &e, Array &u2, Array &v2, Array &e2,
            double g, double h, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate stage 3 of the RK time stepper
   *
   * u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
   *
   */
  // u: elevation x gradient
  auto rhs_u3 = [dt, g, dx_inv](auto tuple) {
    auto [e2, u2, out] = tuple;
    auto dedx = (e2(1, 0) - e2(0, 0)) * dx_inv;
    out(0, 0) *= 1.0 / 3;
    out(0, 0) += 2.0 / 3 * (u2(0, 0) - dt * g * dedx);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0) - 1),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    auto u2_view = dr::mhp::views::submdspan(u2.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    dr::mhp::stencil_for_each(rhs_u3, e2_view, u2_view, u_view);
    dr::mhp::halo(u).exchange();
  }

  // v: elevation y gradient
  auto rhs_v3 = [dt, g, dy_inv](auto tuple) {
    auto [e2, v2, out] = tuple;
    auto dedy = (e2(0, 0) - e2(0, -1)) * dy_inv;
    out(0, 0) *= 1.0 / 3;
    out(0, 0) += 2.0 / 3 * (v2(0, 0) - dt * g * dedy);
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(e.mdspan().extent(0)),
        static_cast<std::size_t>(e.mdspan().extent(1))};
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    auto v2_view = dr::mhp::views::submdspan(v2.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    dr::mhp::stencil_for_each(rhs_v3, e2_view, v2_view, v_view);
    dr::mhp::halo(v).exchange();
  }

  // e: divergence of (u, v)
  auto rhs_e3 = [dt, h, dx_inv, dy_inv](auto tuple) {
    auto [e2, u2, v2, out] = tuple;
    auto dudx = (u2(0, 0) - u2(-1, 0)) * dx_inv;
    auto dvdy = (v2(0, 1) - v2(0, 0)) * dy_inv;
    out(0, 0) *= 1.0 / 3;
    out(0, 0) += 2.0 / 3 * (e2(0, 0) - dt * h * (dudx + dvdy));
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{
        static_cast<std::size_t>(u.mdspan().extent(0)),
        static_cast<std::size_t>(u.mdspan().extent(1))};
    auto e2_view = dr::mhp::views::submdspan(e2.view(), start, end);
    auto u2_view = dr::mhp::views::submdspan(u2.view(), start, end);
    auto v2_view = dr::mhp::views::submdspan(v2.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    dr::mhp::stencil_for_each(rhs_e3, e2_view, u2_view, v2_view, e_view);
    dr::mhp::halo(e).exchange();
  }
};

int run(
    int n, bool benchmark_mode, bool fused_kernels,
    std::function<void()> iter_callback = []() {}) {

  // Arakava C grid
  //
  // T points at cell centers
  // U points at center of x edges
  // V points at center of y edges
  // F points at vertices
  //
  //   |       |       |       |       |
  //   f---v---f---v---f---v---f---v---f-
  //   |       |       |       |       |
  //   u   t   u   t   u   t   u   t   u
  //   |       |       |       |       |
  //   f---v---f---v---f---v---f---v---f-

  // number of cells in x, y direction
  std::size_t nx = n;
  std::size_t ny = n;
  const double xmin = -1, xmax = 1;
  const double ymin = -1, ymax = 1;
  const double lx = xmax - xmin;
  const double ly = ymax - ymin;
  const double dx = lx / nx;
  const double dy = ly / ny;
  const double dx_inv = 1.0 / dx;
  const double dy_inv = 1.0 / dy;
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

  // compute time step
  double t_end = 1.0;
  double t_export = 0.02;

  double c = std::sqrt(g * h);
  double alpha = 0.5;
  double dt = alpha * dx / c;
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

  // initial condition for elevation
  for (auto segment : dr::ranges::segments(e)) {
    if (dr::ranges::rank(segment) == std::size_t(comm_rank)) {
      auto origin = segment.origin();
      auto e = segment.mdspan();

      for (std::size_t i = 0; i < e.extent(0); i++) {
        std::size_t global_i = i + origin[0];
        if (global_i > 0) {
          for (std::size_t j = 0; j < e.extent(1); j++) {
            std::size_t global_j = j + origin[1];
            T x = xmin + dx / 2 + (global_i - 1) * dx;
            T y = ymin + dy / 2 + global_j * dy;
            e(i, j) = initial_elev(x, y, lx, ly);
          }
        }
      }
    }
  }
  dr::mhp::halo(e).exchange();

  auto add = [](auto ops) { return ops.first + ops.second; };
  auto max = [](double x, double y) { return std::max(x, y); };
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
  double initial_v = 0.0;
  auto tic = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < nt + 1; i++) {
    t = i * dt;

    if (t >= next_t_export - 1e-8) {

      double elev_max = dr::mhp::reduce(e, static_cast<T>(0), max);
      double u_max = dr::mhp::reduce(u, static_cast<T>(0), max);

      double total_v =
          (dr::mhp::reduce(e, static_cast<T>(0), std::plus{}) + h) * dx * dy;
      if (i == 0) {
        initial_v = total_v;
      }
      double diff_v = total_v - initial_v;

      if (comm_rank == 0) {
        printf("%2lu %4lu %.3f ", i_export, i, t);
        printf("elev=%7.5f ", elev_max);
        printf("u=%7.5f ", u_max);
        printf("dV=% 6.3e ", diff_v);
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
    }

    // step
    iter_callback();
    if (fused_kernels) {
      stage1(u, v, e, u1, v1, e1, g, h, dx_inv, dy_inv, dt);
      stage2(u, v, e, u1, v1, e1, u2, v2, e2, g, h, dx_inv, dy_inv, dt);
      stage3(u, v, e, u2, v2, e2, g, h, dx_inv, dy_inv, dt);
    } else {
      // RK stage 1: u1 = u + dt*rhs(u)
      rhs(u, v, e, dudt, dvdt, dedt, g, h, dx_inv, dy_inv, dt);
      dr::mhp::transform(dr::mhp::views::zip(u, dudt), u1.begin(), add);
      dr::mhp::transform(dr::mhp::views::zip(v, dvdt), v1.begin(), add);
      dr::mhp::transform(dr::mhp::views::zip(e, dedt), e1.begin(), add);
      dr::mhp::halo(u1).exchange();
      dr::mhp::halo(v1).exchange();
      dr::mhp::halo(e1).exchange();

      // RK stage 2: u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
      rhs(u1, v1, e1, dudt, dvdt, dedt, g, h, dx_inv, dy_inv, dt);
      dr::mhp::transform(dr::mhp::views::zip(u, u1, dudt), u2.begin(),
                         rk_update2);
      dr::mhp::transform(dr::mhp::views::zip(v, v1, dvdt), v2.begin(),
                         rk_update2);
      dr::mhp::transform(dr::mhp::views::zip(e, e1, dedt), e2.begin(),
                         rk_update2);
      dr::mhp::halo(u2).exchange();
      dr::mhp::halo(v2).exchange();
      dr::mhp::halo(e2).exchange();

      // RK stage 3: u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
      rhs(u2, v2, e2, dudt, dvdt, dedt, g, h, dx_inv, dy_inv, dt);
      dr::mhp::transform(dr::mhp::views::zip(u, u2, dudt), u.begin(),
                         rk_update3);
      dr::mhp::transform(dr::mhp::views::zip(v, v2, dvdt), v.begin(),
                         rk_update3);
      dr::mhp::transform(dr::mhp::views::zip(e, e2, dedt), e.begin(),
                         rk_update3);
      dr::mhp::halo(u).exchange();
      dr::mhp::halo(e).exchange();
      dr::mhp::halo(v).exchange();
    }
  }
  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    double t_cpu = duration.count();
    double t_step = t_cpu / nt;
    double read_bw = double(nread) / t_step / (1024 * 1024 * 1024);
    double write_bw = double(nwrite) / t_step / (1024 * 1024 * 1024);
    double flop_rate = double(nflop) / t_step / (1000 * 1000 * 1000);
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
  }

  // Compute error against exact solution
  Array e_exact({nx + 1, ny}, dist);
  dr::mhp::fill(e_exact, 0.0);
  Array error({nx + 1, ny}, dist);
  // initial condition for elevation
  for (auto segment : dr::ranges::segments(e_exact)) {
    if (dr::ranges::rank(segment) == std::size_t(comm_rank)) {
      auto origin = segment.origin();
      auto e = segment.mdspan();

      for (std::size_t i = 0; i < e.extent(0); i++) {
        std::size_t global_i = i + origin[0];
        if (global_i > 0) {
          for (std::size_t j = 0; j < e.extent(1); j++) {
            std::size_t global_j = j + origin[1];
            T x = xmin + dx / 2 + (global_i - 1) * dx;
            T y = ymin + dy / 2 + global_j * dy;
            e(i, j) = exact_elev(x, y, t, lx, ly);
          }
        }
      }
    }
  }
  dr::mhp::halo(e_exact).exchange();
  auto error_kernel = [](auto ops) {
    auto err = ops.first - ops.second;
    return err * err;
  };
  dr::mhp::transform(dr::mhp::views::zip(e, e_exact), error.begin(),
                     error_kernel);
  double err_L2 = dr::mhp::reduce(error, static_cast<T>(0), std::plus{}) * dx *
                  dy / lx / ly;
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
  } else if (nx == 128 && ny == 128) {
    double expected_L2 = 0.007224068445111;
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
  if (comm_rank == 0) {
    std::cout << "SUCCESS" << std::endl;
  }

  return 0;
}

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
    dr::mhp::init(q);
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

  auto error = run(n, benchmark_mode, fused_kernels);
  dr::mhp::finalize();
  MPI_Finalize();
  return error;
}

#else

static void WaveEquation_DR(benchmark::State &state) {

  int n = 4000;
  std::size_t nread, nwrite, nflop;
  calculate_complexity(n, n, nread, nwrite, nflop);
  Stats stats(state, nread, nwrite, nflop);

  auto iter_callback = [&stats]() { stats.rep(); };
  for (auto _ : state) {
    run(n, true, true, iter_callback);
  }
}

DR_BENCHMARK(WaveEquation_DR);

#endif
