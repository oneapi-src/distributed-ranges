// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mp.hpp"
#include "mpi.h"

inline void barrier() { dr::mp::barrier(); }
inline void fence() { dr::mp::fence(); }
inline void fence_on(auto &&obj) { obj.fence(); }

#include <chrono>
#include <memory>
#include <iomanip>

struct MPI_data {
  MPI_Comm comm;
  int rank;
  int size;

  bool host() {
    return rank == 0;
  }
};

static MPI_data mpi_data;

struct Options {
  std::size_t width;
  std::size_t height;
  std::size_t steps;
  std::size_t redundancy;
  bool debug;

  std::unique_ptr<std::ofstream> logfile;

  bool sycl;
  bool device_memory;
};

namespace GameOfLife {

using T = int;
using Array = dr::mp::distributed_mdarray<T, 2>;

void init(std::size_t n, Array& out) {
  std::vector<std::vector<int>> in(4, std::vector<int>(4, 0));
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
  std::vector<int> local(n * 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      local[i * n + j] = in[i][j];
    }
  }
  dr::mp::copy(local.begin(), local.end(), out.begin());
}

void run(std::size_t n, std::size_t m, std::size_t redundancy, std::size_t steps, bool debug) {
  if (mpi_data.host()) {
    std::cout << "Using backend: dr" << std::endl;
    std::cout << "Grid size: " << n << " x " << m << std::endl;
    std::cout << "Time steps:" << steps << std::endl;
    std::cout << "Redundancy " << redundancy << std::endl;
    std::cout << std::endl;
  }

  // construct grid
  auto dist = dr::mp::distribution().halo(1).redundancy(redundancy);
  Array array({n, m}, dist);
  Array array_out({n, m}, dist);
  dr::mp::fill(array, 0);

  init(m, array);

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

  auto print = [n, m](const auto &v) {
      std::vector<int> local(n * n);
      dr::mp::copy(0, v, local.begin());
      if (mpi_data.host()) {
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < m; j++) {
            fmt::print("{}", local[i * m + j] == 1 ? '#' : '.');
          }
          fmt::print("\n");
        }
      }
    };

  std::chrono::duration<double> exchange_duration;
  std::size_t exchange_count = 0;

  auto tic = std::chrono::steady_clock::now();
  for (std::size_t i = 0, next_treshold = 0; i < steps; i++) {
    if (i >= next_treshold && mpi_data.host()) {
      next_treshold += round(static_cast<double>(steps / 100));
      double percent = round(static_cast<double>(i) * 100 / static_cast<double>(steps));
      fmt::print("Steps done {}% ({} of {} steps)\n", percent, i, steps);
    }

    // step
    stencil_for_each_extended<2>(calculate, {1, 1}, {1, 1}, array, array_out);
    stencil_for_each_extended<2>(assign, {0, 0}, {0, 0}, array, array_out);

    // phase with communication - once after (redundancy - 1) steps without communication
    if ((i + 1) % redundancy == 0) {
      if (debug && mpi_data.host()) {
        fmt::print("Exchange at step {}\n", i);
      }
      auto exchange_tic = std::chrono::steady_clock::now();
      array.halo().exchange();
      auto exchange_toc = std::chrono::steady_clock::now();
      exchange_duration += exchange_toc - exchange_tic;
      exchange_count++;

      // Array_out is a temporary, no need to exchange it
    }

    // debug print
    if (debug) {
      if (mpi_data.host()) {
        fmt::print("Array {}:\n", i);
      }
      // print needs a synchronication accros MPI boundary (dr::mp::copy), each node has to execute it
      print(array);
    }
  }
  auto toc = std::chrono::steady_clock::now();

  std::chrono::duration<double> duration = toc - tic;

  if (mpi_data.host()) {
    double t_cpu = duration.count();
    double t_exch = exchange_duration.count();
    double t_step = t_cpu / static_cast<double>(steps);
    double t_exch_step = t_exch / static_cast<double>(exchange_count);

    fmt::print("Steps done 100% ({} of {} steps)\n", steps, steps);
    fmt::print("Duration {} s, including exchange total time {} s\n", t_cpu, t_exch);
    fmt::print("Time per step {} ms\n", t_step * 1000);
    fmt::print("Time per exchange {} ms\n", t_exch_step * 1000);
  }
}

} // namespace GameOfLife

// Initialization functions

void init_MPI(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  mpi_data.comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_data.comm, &mpi_data.rank);
  MPI_Comm_size(mpi_data.comm, &mpi_data.size);

  dr::drlog.debug("MPI: rank = {}, size = {}\n", mpi_data.rank, mpi_data.size);
}

Options parse_options(int argc, char *argv[]) {
  Options out;

  cxxopts::Options options_spec(argv[0], "game of life");

  // clang-format off
  options_spec.add_options()
    ("drhelp", "Print help")
    ("log", "Enable logging")
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
    ("log-filter", "Filter the log", cxxopts::value<std::vector<std::string>>())
    ("device-memory", "Use device memory")
    ("sycl", "Execute on SYCL device")
    ("d,debug", "enable debug logging")
    ("n,size", "Grid width", cxxopts::value<std::size_t>()->default_value("128"))
    ("m,height", "Grid height", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,steps", "Run a fixed number of time steps.", cxxopts::value<std::size_t>()->default_value("100"))
    ("r,redundancy", "Set outer-grid redundancy parameter.", cxxopts::value<std::size_t>()->default_value("2"));
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  out.sycl = options.count("sycl") != 0;
  out.device_memory = options.count("device-memory") != 0;

  if (options.count("drhelp")) {
    std::cout << options_spec.help() << "\n";
    exit(0);
  }

  if (options.count("log")) {
    out.logfile.reset(new std::ofstream(options["logprefix"].as<std::string>() +
                                    fmt::format(".{}.log", mpi_data.rank)));
    dr::drlog.set_file(*out.logfile);
    if (options.count("log-filter")) {
      dr::drlog.filter(options["log-filter"].as<std::vector<std::string>>());
    }
  }

  out.width = options["n"].as<std::size_t>();
  out.height = options.count("m") != 0 ? options["m"].as<std::size_t>() : out.width;
  out.redundancy = options["r"].as<std::size_t>();
  out.steps = options["t"].as<std::size_t>();

  out.debug = options.count("debug") != 0;

  return out;
}

void dr_init(const Options& options) {
#ifdef SYCL_LANGUAGE_VERSION
  if (options.sycl) {
    sycl::queue q;
    fmt::print("Running on sycl device: {}, memory: {}\n", q.get_device().get_info<sycl::info::device::name>(), options.device_memory ? "device" : "shared");
    dr::mp::init(q, options.device_memory ? sycl::usm::alloc::device
                                          : sycl::usm::alloc::shared);
    return;
  }
#endif
  fmt::print("Running on CPU\n");
  dr::mp::init();
}

// Main loop

int main(int argc, char *argv[]) {
  init_MPI(argc, argv);
  Options options = parse_options(argc, argv);
  dr_init(options);

  GameOfLife::run(options.width, options.height, options.redundancy, options.steps, options.debug);

  dr::mp::finalize();
  MPI_Finalize();

  return 0;
}
