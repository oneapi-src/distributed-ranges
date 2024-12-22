// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mp.hpp"
#include "mpi.h"
#include <chrono>
#include <memory>
#include <iomanip>

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

namespace GameOfLife {

using T = int;
using Array = dr::mp::distributed_mdarray<T, 2>;

void init(std::size_t n, Array& out) {
  std::vector<std::vector<int>> in(n, std::vector<int>(n, 0));
  /*
    1 0 0
    0 1 1
    1 1 0
   */
  // clang-format off
  in[1][1] = 1; in[1][2] = 0; in[1][3] = 0;
  in[2][1] = 0; in[2][2] = 1; in[2][3] = 1;
  in[3][1] = 1; in[3][2] = 1; in[3][3] = 0;
  // clang-format on
  std::vector<int> local(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      local[i * n + j] = in[i][j];
    }
  }
  dr::mp::copy(local.begin(), local.end(), out.begin());
}

void run(std::size_t n, std::size_t redundancy, std::size_t steps, bool debug) {
  if (comm_rank == 0) {
    std::cout << "Using backend: dr" << std::endl;
    std::cout << "Grid size: " << n << " x " << n << std::endl;
    std::cout << "Time steps:" << steps << std::endl;
    std::cout << "Redundancy " << redundancy << std::endl;
    std::cout << std::endl;
  }

  // construct grid
  auto dist = dr::mp::distribution().halo(1).redundancy(redundancy);
  Array array({n, n}, dist);
  Array array_out({n, n}, dist);
  dr::mp::fill(array, 0);
  dr::mp::fill(array_out, 0);

  init(n, array);

  // execute one calculation for one cell in game of life
  auto calculate = [](auto stencils) {
      auto [x, x_out] = stencils;
      // because below we calculate the sum of all 9 cells,
      // but we want the output only of 8 neighbourhs, subtract the value of self.
      int live_neighbours = -x(0, 0);
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          live_neighbours += x(i, j); // alive == 1, dead == 0, so simple addition works
        }
      }

      if (x(0, 0) == 1) { // self if alive
        if (live_neighbours == 2 || live_neighbours == 3) {
          x_out(0, 0) = 1;
        } else {
          x_out(0, 0) = 0;
        }
      }
      else { // self is dead
        if (live_neighbours == 3) {
          x_out(0, 0) = 1;
        } else {
          x_out(0, 0) = 0;
        }
      }
    };

  // assign values of second array to first array
  auto assign = [](auto stencils) {
      auto [x, x_out] = stencils;
      x(0, 0) = x_out(0, 0);
    };

  auto tic = std::chrono::steady_clock::now();

  auto print = [n](const auto &v) {
      std::vector<int> local(n * n);
      copy(v, local.begin());
      if (comm_rank == 0) {
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            std::cout << local[i * n + j] << " ";
          }
          std::cout << "\n";
        }
      }
    };

  for (std::size_t i = 0; i < steps; i++) {
    if (comm_rank == 0) {
      std::cout << "Step " << i << "\n";
    }
    // step
    stencil_for_each_extended<2>(calculate, {1, 1}, {1, 1}, array, array_out);
    stencil_for_each_extended<2>(assign, {0, 0}, {0, 0}, array, array_out);
    // phase with communication - once after (redundancy - 1) steps without communication
    if ((i + 1) % redundancy == 0) {
      if (comm_rank == 0) {
        std::cout << "Exchange\n";
      }
      array.halo().exchange();
      // Array_out is a temporary, no need to exchange it
    }
    if (debug) {
      if (comm_rank == 0) {
        std::cout << "Array " << i << ":\n";
      }
      print(array);
      if (comm_rank == 0) {
        std::cout << "\n";
      }
    }
  }

  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    double t_cpu = duration.count();
    double t_step = t_cpu / static_cast<double>(steps);
    std::cout << "Duration: " << std::setprecision(3) << t_cpu << " s" << std::endl;
    std::cout << "Time per step: " << std::setprecision(2) << t_step * 1000 << " ms" << std::endl;
  }
}

} // namespace GameOfLife

#ifdef STANDALONE_BENCHMARK

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  cxxopts::Options options_spec(argv[0], "game of life");
  // clang-format off
  options_spec.add_options()
    ("n,size", "Grid size", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,steps", "Run a fixed number of time steps.", cxxopts::value<std::size_t>()->default_value("100"))
    ("r,redundancy", "Set outer-grid redundancy parameter.", cxxopts::value<std::size_t>()->default_value("2"))
    ("sycl", "Execute on SYCL device")
    ("l,log", "enable logging")
    ("d,debug", "enable debug logging")
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

  bool debug = false;
  if (options.count("debug")) {
    debug = true;
  }

  GameOfLife::run(n, redundancy, steps, debug);
  dr::mp::finalize();
  MPI_Finalize();
  return 0;
}

#else

static void GameOfLife_DR(benchmark::State &state) {}

DR_BENCHMARK(GameOfLife_DR);

#endif
