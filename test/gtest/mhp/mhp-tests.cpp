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
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
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
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
    ("log-filter", "Filter the log", cxxopts::value<std::vector<std::string>>())
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

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile.reset(new std::ofstream(options["logprefix"].as<std::string>() +
                                    fmt::format(".{}.log", comm_rank)));
    dr::drlog.set_file(*logfile);
    if (options.count("log-filter")) {
      dr::drlog.filter(options["log-filter"].as<std::vector<std::string>>());
    }
  }

  dr_init();
  dr::drlog.debug("Rank: {}\n", comm_rank);

  auto res = RUN_ALL_TESTS();

  dr::mhp::finalize();
  MPI_Finalize();

  return res;
}
