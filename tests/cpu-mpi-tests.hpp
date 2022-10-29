#include <gtest/gtest.h>

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

inline void expect_eq(auto &r1, auto &r2, int root = comm_rank) {
  if (comm_rank == root) {
    for (size_t i = 0; i < r1.size(); i++) {
      EXPECT_EQ(r1[i], r2[i]);
    }
  }
}
