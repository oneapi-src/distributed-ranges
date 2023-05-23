// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;

auto stencil_op = [](auto &&v) {
  auto p = &v;
  return p[-1] + p[0] + p[+1];
};

int check(auto dv, auto n, auto steps) {
  // Serial stencil
  std::vector<int> a(n), b(n);
  rng::iota(a, 100);
  rng::fill(b, 0);

  auto in = rng::subrange(a.begin() + 1, a.end() - 1);
  auto out = rng::subrange(b.begin() + 1, b.end() - 1);
  for (std::size_t s = 0; s < steps; s++) {
    rng::transform(in, out.begin(), stencil_op);
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
  dr::mhp::halo_bounds hb(1);
  dr::mhp::distributed_vector<int> a(n, hb), b(n, hb);
  dr::mhp::iota(a, 100);
  dr::mhp::fill(b, 0);

  auto in = rng::subrange(a.begin() + 1, a.end() - 1);
  auto out = rng::subrange(b.begin() + 1, b.end() - 1);
  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    dr::mhp::transform(in, out.begin(), stencil_op);
    std::swap(in, out);
  }

  if (comm_rank == 0) {
    return check(in, n, steps);
  } else {
    return 0;
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  dr::mhp::init();

  cxxopts::Options options_spec(argv[0], "stencil 1d");
  // clang-format off
  options_spec.add_options()
    ("n", "Size of array", cxxopts::value<std::size_t>()->default_value("10"))
    ("s", "Number of time steps", cxxopts::value<std::size_t>()->default_value("5"))
    ("help", "Print help");
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  auto error =
      stencil(options["n"].as<std::size_t>(), options["s"].as<std::size_t>());

  MPI_Finalize();
  return error;
}
