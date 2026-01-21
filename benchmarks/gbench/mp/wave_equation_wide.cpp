// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mp.hpp"
#include "mpi.h"
#include "wave_utils.hpp"
#include <chrono>
#include <iomanip>
#include <memory>

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

namespace WaveEquation {

using T = double;
using Array = dr::mp::distributed_mdarray<T, 2>;

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
  double sol_x = std::cos(2.0 * M_PI * x / lx);
  double sol_y = std::cos(2.0 * M_PI * y / ly);
  double omega = c * M_PI * std::hypot(1.0 / lx, 1.0 / ly);
  double sol_t = std::cos(2.0 * omega * t);
  return amp * sol_x * sol_y * sol_t;
}

double initial_elev(double x, double y, double lx, double ly) {
  return exact_elev(x, y, 0.0, lx, ly);
}

void rhs(Array &u, Array &v, Array &e, Array &dudt, Array &dvdt, Array &dedt,
         double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate right hand side of the equations
   */
  auto rhs_dedx = [dt, dx_inv](auto v) {
    auto [in, out] = v;
    out(0, 0) = -dt * g * (in(1, 0) - in(0, 0)) * dx_inv;
  };
  stencil_for_each_extended<2>(rhs_dedx, {0, 0}, {1, 0}, e, dudt);

  auto rhs_dedy = [dt, dy_inv](auto v) {
    auto [in, out] = v;
    out(0, 0) = -dt * g * (in(0, 0) - in(0, -1)) * dy_inv;
  };
  stencil_for_each_extended<2>(rhs_dedy, {0, 1}, {0, 0}, e, dvdt);

  auto rhs_div = [dt, dx_inv, dy_inv](auto args) {
    auto [u, v, out] = args;
    auto dudx = (u(0, 0) - u(-1, 0)) * dx_inv;
    auto dvdy = (v(0, 1) - v(0, 0)) * dy_inv;
    out(0, 0) = -dt * h * (dudx + dvdy);
  };
  stencil_for_each_extended<2>(rhs_div, {1, 0}, {0, 0}, u, v, dedt);
}

int run(
    std::size_t n, std::size_t redundancy, std::size_t steps,
    std::function<void()> iter_callback = []() {}) {
  // construct grid
  // number of cells in x, y direction
  std::size_t nx = n;
  std::size_t ny = n;
  const double xmin = -1, xmax = 1;
  const double ymin = -1, ymax = 1;
  ArakawaCGrid grid(xmin, xmax, ymin, ymax, nx, ny);

  auto dist = dr::mp::distribution().halo(1).redundancy(redundancy);

  // statistics
  std::size_t nread, nwrite, nflop;
  calculate_complexity(nx, ny, nread, nwrite, nflop);

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
  double c = std::sqrt(g * h);
  double alpha = 0.5;
  double dt = alpha * std::min(grid.dx, grid.dy) / c;
  std::size_t nt = steps;
  dt = 1e-5;
  double t_end = static_cast<double>(nt) * dt;
  double t_export = 25 * dt;

  if (comm_rank == 0) {
    std::cout << "Time step: " << dt << " s" << std::endl;
    std::cout << "Total run time: " << std::fixed << std::setprecision(1);
    std::cout << t_end << " s, ";
    std::cout << nt << " time steps" << std::endl;
    std::cout << "Redundancy " << redundancy << std::endl;
  }

  std::cout << "before e\n";
  // state variables
  // water elevation at T points
  Array e({nx + 1, ny}, dist);
  std::cout << "after e\n";
  dr::mp::fill(e, 0.0);
  std::cout << "after fill e\n";
  // x velocity at U points
  Array u({nx + 1, ny}, dist);
  dr::mp::fill(u, 0.0);
  // y velocity at V points
  Array v({nx + 1, ny + 1}, dist);
  dr::mp::fill(v, 0.0);

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

  std::cout << "After all arrays\n";

  dr::mp::fill(dedt, 0);
  dr::mp::fill(dudt, 0);
  dr::mp::fill(dvdt, 0);
  std::cout << "After fill\n";

  dr::mp::halo(dedt).exchange();
  dr::mp::halo(dudt).exchange();
  dr::mp::halo(dvdt).exchange();
  std::cout << "After first exchange\n";

  auto init_op = [xmin, ymin, grid](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i > 0) {
      std::size_t global_j = index[1];
      T x = xmin + grid.dx / 2 + static_cast<double>(global_i - 1) * grid.dx;
      T y = ymin + grid.dy / 2 + static_cast<double>(global_j) * grid.dy;
      o = initial_elev(x, y, grid.lx, grid.ly);
    }
  };
  dr::mp::for_each(init_op, e);
  std::cout << "After mp::for_each\n";

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

  // RK stage 1: u1 = u + dt*rhs(u)
  auto stage_1 = [&] {
    rhs(u, v, e, dudt, dvdt, dedt, grid.dx_inv, grid.dy_inv, dt);
    dr::mp::transform(dr::mp::views::zip(u, dudt), u1.begin(), add);
    dr::mp::transform(dr::mp::views::zip(v, dvdt), v1.begin(), add);
    dr::mp::transform(dr::mp::views::zip(e, dedt), e1.begin(), add);
  };
  // RK stage 2: u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
  auto stage_2 = [&] {
    rhs(u1, v1, e1, dudt, dvdt, dedt, grid.dx_inv, grid.dy_inv, dt);
    dr::mp::transform(dr::mp::views::zip(u, u1, dudt), u2.begin(), rk_update2);
    dr::mp::transform(dr::mp::views::zip(v, v1, dvdt), v2.begin(), rk_update2);
    dr::mp::transform(dr::mp::views::zip(e, e1, dedt), e2.begin(), rk_update2);
  };
  // RK stage 3: u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
  auto stage_3 = [&] {
    rhs(u2, v2, e2, dudt, dvdt, dedt, grid.dx_inv, grid.dy_inv, dt);
    dr::mp::transform(dr::mp::views::zip(u, u2, dudt), u.begin(), rk_update3);
    dr::mp::transform(dr::mp::views::zip(v, v2, dvdt), v.begin(), rk_update3);
    dr::mp::transform(dr::mp::views::zip(e, e2, dedt), e.begin(), rk_update3);
  };

  for (std::size_t i = 0; i < nt + 1; i++) {
    std::cout << "i = " << i << "\n";
    t = static_cast<double>(i) * dt;

    if (t >= next_t_export - 1e-8) {

      double elev_max = dr::mp::reduce(e, static_cast<T>(0), max);
      double u_max = dr::mp::reduce(u, static_cast<T>(0), max);

      double total_v = (dr::mp::reduce(e, static_cast<T>(0), std::plus{}) + h) *
                       grid.dx * grid.dy;
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
      next_t_export = static_cast<double>(i_export) * t_export;
    }

    // step
    iter_callback();
    if ((i + 1) % redundancy == 0) {
      // phase with communication - once after (redundancy - 1) steps without
      // communication
      dr::mp::halo(e).exchange();
      dr::mp::halo(u).exchange();
      dr::mp::halo(v).exchange();
      stage_1();

      dr::mp::halo(u1).exchange();
      dr::mp::halo(v1).exchange();
      dr::mp::halo(e1).exchange();
      stage_2();

      dr::mp::halo(u2).exchange();
      dr::mp::halo(v2).exchange();
      dr::mp::halo(e2).exchange();
      stage_3();
    } else {
      // Phase without communication
      stage_1();
      stage_2();
      stage_3();
    }
  }

  dr::mp::halo(e).exchange();
  dr::mp::halo(u).exchange();
  dr::mp::halo(v).exchange();
  dr::mp::halo(u1).exchange();
  dr::mp::halo(v1).exchange();
  dr::mp::halo(e1).exchange();
  dr::mp::halo(u2).exchange();
  dr::mp::halo(v2).exchange();
  dr::mp::halo(e2).exchange();

  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    double t_cpu = duration.count();
    double t_step = t_cpu / static_cast<double>(nt);
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

  // Compute error against exact solution
  Array e_exact({nx + 1, ny}, dist);
  dr::mp::fill(e_exact, 0.0);
  Array error({nx + 1, ny}, dist);

  auto exact_op = [xmin, ymin, grid, t](auto index, auto v) {
    auto &[o] = v;

    std::size_t global_i = index[0];
    if (global_i > 0) {
      std::size_t global_j = index[1];
      T x = xmin + grid.dx / 2 + static_cast<double>(global_i - 1) * grid.dx;
      T y = ymin + grid.dy / 2 + static_cast<double>(global_j) * grid.dy;
      o = exact_elev(x, y, t, grid.lx, grid.ly);
    }
  };
  dr::mp::for_each(exact_op, e_exact);
  dr::mp::halo(e_exact).exchange();
  auto error_kernel = [](auto ops) {
    auto err = ops.first - ops.second;
    return err * err;
  };
  dr::mp::transform(dr::mp::views::zip(e, e_exact), error.begin(),
                    error_kernel);
  double err_L2 = dr::mp::reduce(error, static_cast<T>(0), std::plus{}) *
                  grid.dx * grid.dy / grid.lx / grid.ly;
  err_L2 = std::sqrt(err_L2);
  if (comm_rank == 0) {
    std::cout << "L2 error: " << std::setw(7) << std::scientific;
    std::cout << std::setprecision(5) << err_L2 << std::endl;
  }
  return 0;
}

} // namespace WaveEquation

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
    ("t,steps", "Run a fixed number of time steps.", cxxopts::value<std::size_t>()->default_value("100"))
    ("r,redundancy", "Set outer-grid redundancy parameter.", cxxopts::value<std::size_t>()->default_value("2"))
    ("sycl", "Execute on SYCL device")
    ("l,log", "enable logging")
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
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

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile =
        std::make_unique<std::ofstream>(options["logprefix"].as<std::string>() +
                                        fmt::format(".{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }

  if (options.count("sycl")) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue q = dr::mp::select_queue();
    std::cout << "Run on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    dr::mp::init(q, options.count("device-memory") ? sycl::usm::alloc::device
                                                   : sycl::usm::alloc::shared);
#else
    std::cout << "Sycl support requires icpx\n";
    exit(1);
#endif
  } else {
    if (comm_rank == 0) {
      std::cout << "Run on: CPU\n";
    }
    dr::mp::init();
  }

  std::size_t n = options["n"].as<std::size_t>();
  std::size_t redundancy = options["r"].as<std::size_t>();
  std::size_t steps = options["t"].as<std::size_t>();

  auto error = WaveEquation::run(n, redundancy, steps);
  dr::mp::finalize();
  MPI_Finalize();
  return error;
}

#else

static void WaveEquation_DR(benchmark::State &state) {

  int n = ::sqrtl(default_vector_size);

  // ugly hack to make it working in reasonable time in benchmarking framework
  // drbench.py should specify right size or there should be another size option
  // to use here instead of default_vector_size
  n /= 4;

  std::size_t nread, nwrite, nflop;
  WaveEquation::calculate_complexity(n, n, nread, nwrite, nflop);
  Stats stats(state, nread, nwrite, nflop);

  auto iter_callback = [&stats]() { stats.rep(); };
  for (auto _ : state) {
    WaveEquation::run(n, true, true, iter_callback);
  }
}

DR_BENCHMARK(WaveEquation_DR);

#endif
