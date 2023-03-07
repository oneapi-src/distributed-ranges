// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

// Instantiate MHP-specific configurations for common tests
template <typename T> struct CommonTestConfigBase {
  using V = std::vector<T>;

  static auto iota(auto &&r, auto val) { return mhp::iota(r, val); }
};
template <typename T>
struct CommonTestConfigCPU : public CommonTestConfigBase<T> {
  using DV = mhp::distributed_vector<T>;
  using DVA = mhp::distributed_vector<T, std::allocator<T>>;
  static auto policy() { return std::execution::par_unseq; }
};

#ifdef SYCL_LANGUAGE_VERSION
template <typename T>
struct CommonTestConfigSYCL : public CommonTestConfigBase<T> {
  using DV = mhp::distributed_vector<T, mhp::sycl_shared_allocator<T>>;
  using DVA = DV;
  static auto policy() { return mhp::device_policy(); }
};
#endif

using Common_Types = ::testing::Types<
#ifdef SYCL_LANGUAGE_VERSION
    CommonTestConfigSYCL<int>, CommonTestConfigSYCL<float>,
#endif
    CommonTestConfigCPU<int>, CommonTestConfigCPU<float>>;

using MHP_Types = ::testing::Types<
#ifdef SYCL_LANGUAGE_VERSION
    CommonTestConfigSYCL<int>, CommonTestConfigSYCL<float>,
#endif
    CommonTestConfigCPU<int>, CommonTestConfigCPU<float>>;

// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, DistributedVector, Common_Types);
// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, Drop, Common_Types);
// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, ForEach, Common_Types);
INSTANTIATE_TYPED_TEST_SUITE_P(MHP, Reduce, Common_Types);
INSTANTIATE_TYPED_TEST_SUITE_P(MHP, ReduceMHP, MHP_Types);
// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, Subrange, Common_Types);
// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, Take, Common_Types);
// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, TransformView, Common_Types);
// INSTANTIATE_TYPED_TEST_SUITE_P(MHP, Zip, Common_Types);

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
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
