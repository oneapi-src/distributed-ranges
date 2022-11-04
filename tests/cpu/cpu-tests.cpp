#include "cpu-tests.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

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

  // std::ofstream logfile(fmt::format("dr.{}.log", comm_rank));
  // lib::drlog.set_file(logfile);
  lib::drlog.debug("Rank: {}\n", comm_rank);

  ::testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();

  MPI_Finalize();

  return res;
}
