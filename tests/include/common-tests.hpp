// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "mpi.h"

#include "cxxopts.hpp"

#include "dr/distributed-ranges.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

testing::AssertionResult equal(const rng::range auto &r1,
                               const rng::range auto &r2) {
  if (rng::equal(r1, r2)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure()
           << fmt::format("\n    {}\n    {}\n  ", r1, r2);
  }
}

testing::AssertionResult unary_check(const rng::range auto &in,
                                     const rng::range auto &ref,
                                     const rng::range auto &tst) {
  if (rng::equal(ref, tst)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << fmt::format(
               "\n     in: {}\n    ref: {}\n    tst: {}\n  ", in, ref, tst);
  }
}

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
