// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

#include "common-tests.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

inline void expect_eq(const lib::mdspan_2d auto &m1,
                      const lib::mdspan_2d auto &m2, int root = comm_rank) {
  if (comm_rank == root) {
    EXPECT_TRUE(m1.extents() == m2.extents());
    for (std::size_t i = 0; i < m1.extents().extent(0); i++) {
      for (std::size_t j = 0; j < m1.extents().extent(1); j++) {
        EXPECT_EQ(m1(i, j), m2(i, j));
      }
    }
  }
}

template <typename DR> inline void assert_distributed_range() {
  static_assert(rng::range<DR>);
  static_assert(lib::distributed_contiguous_range<DR>);
}
