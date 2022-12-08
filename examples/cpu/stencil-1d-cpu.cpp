#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/distributed-ranges.hpp"

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
  std::vector<int> v_in(n), v_out(n);
  rng::iota(v_in, 100);

  auto *in = &v_in;
  auto *out = &v_out;
  for (std::size_t s = 0; s < steps; s++) {
    std::transform(in->begin() + 1, in->end() - 1, out->begin() + 1,
                   stencil_op);
    std::swap(in, out);
  }

  // Check the result
  if (!rng::equal(*dv, *in)) {
    fmt::print("Mismatch\n");
    if (n < 100) {
      fmt::print("  v: {}\n", *in);
      fmt::print(" dv: {}\n", *dv);
    }
    return 1;
  }
  return 0;
}

int stencil(auto n, auto steps) {
  lib::stencil<1> s(1);
  lib::distributed_vector<int> dv_in(s, n), dv_out(s, n);
  rng::iota(dv_in, 100);

  auto *in = &dv_in;
  auto *out = &dv_out;
  for (std::size_t s = 0; s < steps; s++) {
    lib::transform(in->begin() + 1, in->end() - 1, out->begin() + 1,
                   stencil_op);
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
