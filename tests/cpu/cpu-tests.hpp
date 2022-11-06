#include <gtest/gtest.h>

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

template <rng::range R1, rng::range R2>
inline void expect_eq(R1 &r1, R2 &r2, int root = comm_rank) {
  if (comm_rank == root) {
    for (size_t i = 0; i < r1.size(); i++) {
      EXPECT_EQ(r1[i], r2[i]);
    }
  }
}

template <typename DR> inline void assert_distributed_range() {
  static_assert(std::random_access_iterator<lib::index_iterator<DR>>);
  static_assert(rng::range<DR>);
  static_assert(lib::distributed_contiguous_range<DR>);
}
