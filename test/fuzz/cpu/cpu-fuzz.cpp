// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-fuzz.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;
int controller_rank;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  MPI_Init(argc, argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  mhp::init();

  cxxopts::Options options_spec((*argv)[0], "Fuzz CPU tests");

  // clang-format off
  options_spec.add_options()
    ("controller", "Rank that controls fuzzing", cxxopts::value<int>()->default_value("0"))
    ("log", "Enable logging")
    ("fuzz_help", "Print help");
  // clang-format on
  options_spec.allow_unrecognised_options();

  try {
    options = options_spec.parse(*argc, *argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("fuzz_help")) {
    std::cout << options_spec.help() << "\n";
    exit(0);
  }

  controller_rank = options["controller"].as<int>();
  if (comm_rank == controller_rank) {
    fmt::print("Controller: {}\n", controller_rank);
  }
  return 0;
}

struct fuzz_spec {
  uint8_t algorithm;
  uint8_t n, b, e;
};

enum class Algorithms {
  Copy,
  Transform,
  Last,
};

extern "C" int LLVMFuzzerTestOneInput(const fuzz_spec *my_spec,
                                      std::size_t size) {
  // Controller broadcasts its fuzz spec
  MPI_Bcast(&size, sizeof(size), MPI_BYTE, controller_rank, comm);
  if (sizeof(fuzz_spec) < size)
    return 0;

  auto spec = *my_spec;
  MPI_Bcast(&spec, sizeof(spec), MPI_BYTE, controller_rank, comm);

  auto n = spec.n;
  auto b = spec.b;
  auto e = spec.e;
  if (n > 64 || b > n || e > n || b > e || n == 0)
    return 0;

  // fmt::print("n: {} b: {} e: {}\n", n, b, e);

  // Algorithm number is 8 bits. Mod it so we don't generate many test
  // cases that do nothing.
  switch (Algorithms(spec.algorithm % std::size_t(Algorithms::Last))) {
  case Algorithms::Copy:
    check_copy(n, b, e);
    break;
  case Algorithms::Transform:
    check_transform(n, b, e);
    break;
  default:
    break;
  }

  return 0;
}
