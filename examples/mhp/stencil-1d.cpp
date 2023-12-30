// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"

auto stencil_op = [](auto &&r) { return r[0] + r[1] + r[2]; };

int check(auto dv, auto n, auto steps) {
  // Serial stencil
  std::vector<int> a(n), b(n);
  rng::iota(a, 100);
  rng::fill(b, 0);

  auto in = rng::ref_view(a);
  auto out = rng::ref_view(b);

  for (std::size_t s = 0; s < steps; s++) {
    rng::transform(rng::views::sliding(in, 3), out.begin() + 1, stencil_op);
    std::swap(in, out);
  }

  // Check the result
  if (!rng::equal(dv, in)) {
    fmt::print("Mismatch\n");
    if (n < 100) {
      fmt::print(" local: {}\n"
                 " dist:: {}\n",
                 in, dv);
    }
    return 1;
  }
  return 0;
}

int stencil(auto n, auto steps) {
  auto dist = dr::mhp::distribution().halo(1);
  dr::mhp::distributed_vector<int> a(n, dist), b(n, dist);
  dr::mhp::iota(a, 100);
  dr::mhp::fill(b, 0);

  auto in = rng::ref_view(a);
  auto out = rng::ref_view(b);

  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    dr::mhp::transform(dr::mhp::views::sliding(in, 3), out.begin() + 1,
                       stencil_op);

    std::swap(in, out);
  }

  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  if (comm_rank == 0) {
    return check(in, n, steps);
  } else {
    return 0;
  }
}

int main(int argc, char *argv[]) {

  cxxopts::Options options_spec(argv[0], "stencil 1d");
  // clang-format off
  options_spec.add_options()
    ("n", "Size of array", cxxopts::value<std::size_t>()->default_value("10"))
    ("s", "Number of time steps", cxxopts::value<std::size_t>()->default_value("5"))
    ("l,log", "enable logging")
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
    ("help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("help")) {
    std::cout << options_spec.help() << "\n";
    return 0;
  }

  MPI_Init(&argc, &argv);

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    logfile.reset(new std::ofstream(options["logprefix"].as<std::string>() +
                                    fmt::format(".{}.log", rank)));
    dr::drlog.set_file(*logfile);
  }

  dr::mhp::init();

  auto error =
      stencil(options["n"].as<std::size_t>(), options["s"].as<std::size_t>());

  dr::mhp::finalize();
  MPI_Finalize();
  return error;
}
