// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;

cxxopts::ParseResult options;

void dr_init() {
#ifdef SYCL_LANGUAGE_VERSION
  if (options.count("sycl")) {
    sycl::queue q;
    if (comm_rank == 0) {
      fmt::print("Enable sycl device: {}\n",
                 q.get_device().get_info<sycl::info::device::name>());
    }
    dr::mhp::init(q, options.count("device-memory") ? sycl::usm::alloc::device
                                                    : sycl::usm::alloc::shared);
    return;
  }
#endif

  if (comm_rank == 0) {
    fmt::print("Enable CPU\n");
  }
  dr::mhp::init();
}

int main(int argc, char *argv[]) {
  comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  comm_rank = rank;
  comm_size = size;

  ::testing::InitGoogleTest(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR MHP tests");

  // clang-format off
  options_spec.add_options()
    ("drhelp", "Print help")
    ("log", "Enable logging")
    ("device-memory", "Use device memory")
    ("sycl", "Execute on SYCL device");
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("drhelp")) {
    std::cout << options_spec.help() << "\n";
    exit(0);
  }

  dr_init();
  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  auto res = RUN_ALL_TESTS();

  if (logfile) {
    delete logfile;
  }

  dr::mhp::finalize();
  MPI_Finalize();

  return res;
}
