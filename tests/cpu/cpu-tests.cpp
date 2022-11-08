#include "cpu-tests.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;

// Demonstrate some basic assertions.
TEST(CpuMpiTests, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  ::testing::InitGoogleTest(&argc, argv);

  cxxopts::Options options_spec(argv[0], "DR CPU tests");

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
