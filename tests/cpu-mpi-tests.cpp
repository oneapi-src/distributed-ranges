#include <gtest/gtest.h>

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

// Demonstrate some basic assertions.
TEST(CpuMpiTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

// Basic functions for distributed_vector
TEST(CpuMpiTest, DISABLED_DistributedVector) {
  auto dist = lib::block_cyclic(lib::partition::div, comm);
  lib::distributed_vector<int, lib::block_cyclic> a(10, dist);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  ::testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();

  MPI_Finalize();

  return res;
}
