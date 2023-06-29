// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include "mpi.h"
#include <chrono>
#include <iomanip>

using T = double;

MPI_Comm comm;
int comm_rank;
int comm_size;

// gravitational acceleration
constexpr double g = 9.81;
// water depth
constexpr double h = 1.0;

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

void rhs(dr::mhp::distributed_dense_matrix<T> &u,
         dr::mhp::distributed_dense_matrix<T> &v,
         dr::mhp::distributed_dense_matrix<T> &e,
         dr::mhp::distributed_dense_matrix<T> &dudx,
         dr::mhp::distributed_dense_matrix<T> &dvdy,
         dr::mhp::distributed_dense_matrix<T> &dudt,
         dr::mhp::distributed_dense_matrix<T> &dvdt,
         dr::mhp::distributed_dense_matrix<T> &dedt, double g, double h,
         double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate right hand side of the equations
   */

  auto upx = dr::mhp::subrange(u, {1, u.shape()[0]}, {0, u.shape()[1]});
  auto vpy = dr::mhp::subrange(v, {1, v.shape()[0]}, {1, v.shape()[1]});
  auto epx = dr::mhp::subrange(e, {1, e.shape()[0] - 1}, {0, e.shape()[1]});
  auto epy = dr::mhp::subrange(e, {1, e.shape()[0]}, {0, e.shape()[1] - 1});

  auto dudx_view =
      dr::mhp::subrange(dudx, {1, dudx.shape()[0]}, {0, dudx.shape()[1]});
  auto dvdy_view =
      dr::mhp::subrange(dvdy, {1, dvdy.shape()[0]}, {0, dvdy.shape()[1]});

  auto dudt_interior =
      dr::mhp::subrange(dudt, {1, dudt.shape()[0] - 1}, {0, dudt.shape()[1]});
  auto dvdt_interior =
      dr::mhp::subrange(dvdt, {1, dvdt.shape()[0]}, {1, dvdt.shape()[1] - 1});

  // -dt * g * d(e)/dx
  auto rhs_dedx = [=](auto &e) {
    return -dt * g * (e[{1, 0}] - e[{0, 0}]) * dx_inv;
  };

  // -dt * g * d(e)/dy
  auto rhs_dedy = [=](auto &e) {
    return -dt * g * (e[{0, 1}] - e[{0, 0}]) * dy_inv;
  };

  // -dt * h * d(u)/dx
  auto rhs_dudx = [=](auto &e) {
    return -dt * h * (e[{0, 0}] - e[{-1, 0}]) * dx_inv;
  };

  // -dt * h * d(u)/dx
  auto rhs_dvdy = [=](auto &e) {
    return -dt * h * (e[{0, 0}] - e[{0, -1}]) * dy_inv;
  };

  // -dt * h * (d(u)/dx + d(v)/dy)
  // auto rhs_divhuv = [=](auto &ops) {
  //   auto u = ops.first;
  //   auto v = ops.second;
  //   auto dudx = (u[{0, 0}] - u[{0, -1}])*dx_inv;
  //   auto dvdx = (v[{0, 0}] - v[{-1, 0}])*dy_inv;
  //   return -dt*h*(dudx + dvdx);
  // };

  auto add = [](auto ops) { return ops.first + ops.second; };

  dr::mhp::transform(epx, dudt_interior.begin(), rhs_dedx);
  dr::mhp::transform(epy, dvdt_interior.begin(), rhs_dedy);
  dr::mhp::transform(upx, dudx_view.begin(), rhs_dudx);
  dr::mhp::transform(vpy, dvdy_view.begin(), rhs_dvdy);
  dr::mhp::transform(dr::mhp::views::zip(dudx, dvdy), dedt.begin(), add);

  // FIXME merged kernel does not work
  // dr::mhp::transform(dr::mhp::views::zip(upx, vpy), dedt.begin(),
  // rhs_divhuv);
};

int run(int n, bool benchmark_mode) {

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
  // dr::mhp::halo_bounds hb(1);
  auto dist = dr::mhp::distribution().halo(1);

  if (comm_rank == 0) {
    std::cout << "Using backend: dr" << std::endl;
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
  dr::mhp::distributed_dense_matrix<T> e(nx + 1, ny, 0, dist);
  // x velocity at U points
  dr::mhp::distributed_dense_matrix<T> u(nx + 1, ny, 0, dist);
  // y velocity at V points
  dr::mhp::distributed_dense_matrix<T> v(nx + 1, ny + 1, 0, dist);

  // state for RK stages
  dr::mhp::distributed_dense_matrix<T> e1(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> u1(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> v1(nx + 1, ny + 1, 0, dist);
  dr::mhp::distributed_dense_matrix<T> e2(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> u2(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> v2(nx + 1, ny + 1, 0, dist);

  // time tendencies
  dr::mhp::distributed_dense_matrix<T> dedt(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> dudt(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> dvdt(nx + 1, ny + 1, 0, dist);

  // temporary arrays
  // FIXME these should not be necessary
  dr::mhp::distributed_dense_matrix<T> dudx(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> dvdy(nx + 1, ny, 0, dist);

  // initial condition for elevation
  // TODO cleaner interface
  for (auto r : e.rows()) {
    if (r.segment()->is_local()) {
      auto i = r.idx();
      std::size_t j = 0;
      for (auto &v : r) {
        T x = xmin + dx / 2 + (i - 1) * dx;
        T y = ymin + dy / 2 + j * dy;
        if (i > 0) {
          v = initial_elev(x, y, lx, ly);
        } else {
          v = 0;
        }
        j++;
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
  double initial_v;
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
    // RK stage 1: u1 = u + dt*rhs(u)
    rhs(u, v, e, dudx, dvdy, dudt, dvdt, dedt, g, h, dx_inv, dy_inv, dt);
    dr::mhp::transform(dr::mhp::views::zip(u, dudt), u1.begin(), add);
    dr::mhp::transform(dr::mhp::views::zip(v, dvdt), v1.begin(), add);
    dr::mhp::transform(dr::mhp::views::zip(e, dedt), e1.begin(), add);
    dr::mhp::halo(u1).exchange();
    dr::mhp::halo(v1).exchange();
    dr::mhp::halo(e1).exchange();

    // RK stage 2: u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
    rhs(u1, v1, e1, dudx, dvdy, dudt, dvdt, dedt, g, h, dx_inv, dy_inv, dt);
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
    rhs(u2, v2, e2, dudx, dvdy, dudt, dvdt, dedt, g, h, dx_inv, dy_inv, dt);
    dr::mhp::transform(dr::mhp::views::zip(u, u2, dudt), u.begin(), rk_update3);
    dr::mhp::transform(dr::mhp::views::zip(v, v2, dvdt), v.begin(), rk_update3);
    dr::mhp::transform(dr::mhp::views::zip(e, e2, dedt), e.begin(), rk_update3);
    dr::mhp::halo(u).exchange();
    dr::mhp::halo(v).exchange();
    dr::mhp::halo(e).exchange();
  }
  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    std::cout << "Duration: " << std::setprecision(2) << duration.count();
    std::cout << " s" << std::endl;
  }

  // Compute error against exact solution
  dr::mhp::distributed_dense_matrix<T> e_exact(nx + 1, ny, 0, dist);
  dr::mhp::distributed_dense_matrix<T> error(nx + 1, ny, 0, dist);
  for (auto r : e_exact.rows()) {
    if (r.segment()->is_local()) {
      auto i = r.idx();
      std::size_t j = 0;
      for (auto &v : r) {
        T x = xmin + dx / 2 + (i - 1) * dx;
        T y = ymin + dy / 2 + j * dy;
        if (i > 0) {
          v = exact_elev(x, y, t, lx, ly);
        } else {
          v = 0;
        }
        j++;
      }
    }
  }
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
    if (fabs(rel_err) > rel_tolerance) {
      if (comm_rank == 0) {
        std::cout << "ERROR: L2 error deviates from reference value: "
                  << expected_L2 << ", relative error: " << rel_err
                  << std::endl;
      }
      return 1;
    }
  } else {
    double tolerance = 1e-2;
    if (err_L2 > tolerance) {
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

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  dr::mhp::init();

  cxxopts::Options options_spec(argv[0], "wave equation");
  // clang-format off
  options_spec.add_options()
    ("n", "Grid size", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,benchmark-mode", "Run a fixed number of time steps.", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  std::size_t n = options["n"].as<std::size_t>();
  bool benchmark_mode = options["t"].as<bool>();

  auto error = run(n, benchmark_mode);
  MPI_Finalize();
  return error;
}
