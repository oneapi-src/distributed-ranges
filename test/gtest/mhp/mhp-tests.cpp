// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

// Use this for shorter build time
// #define MINIMAL_TEST 1
#ifdef MINIMAL_TEST

using AllTypes = ::testing::Types<mhp::distributed_vector<int>>;
#include "common/copy.hpp"

#else

using AllTypes = ::testing::Types<
#ifdef SYCL_LANGUAGE_VERSION
    mhp::distributed_vector<int, mhp::sycl_shared_allocator<int>>,
    mhp::distributed_vector<float, mhp::sycl_shared_allocator<float>>,
#endif
    mhp::distributed_vector<int>, mhp::distributed_vector<float>>;

using CPUTypes = ::testing::Types<mhp::distributed_vector<int>,
                                  mhp::distributed_vector<float>>;

#include "common/all.hpp"
#include "common/copy.hpp"
#include "common/distributed_vector.hpp"
#include "common/drop.hpp"
#include "common/fill.hpp"
#include "common/for_each.hpp"
#include "common/reduce.hpp"
#include "common/subrange.hpp"
// Fails with everyting but g++12
// #include "common/take.hpp"
#include "common/transform_view.hpp"
#include "common/zip.hpp"

#include "reduce.hpp"

#endif

MPI_Comm comm;
std::size_t comm_rank;
std::size_t comm_size;

cxxopts::ParseResult options;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  comm_rank = rank;
  comm_size = size;
  mhp::init();

  ::testing::InitGoogleTest(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR MHP tests");

  // clang-format off
  options_spec.add_options()
    ("log", "Enable logging")
    ("drhelp", "Print help");
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

  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    lib::drlog.set_file(*logfile);
  }
  lib::drlog.debug("Rank: {}\n", comm_rank);

  auto res = RUN_ALL_TESTS();

  if (logfile) {
    delete logfile;
  }

  MPI_Finalize();

  return res;
}
