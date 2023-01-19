// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

extern void check_copy(std::size_t n, std::size_t b, std::size_t e);
extern void check_transform(std::size_t n, std::size_t b, std::size_t e);

bool is_equal(const rng::range auto &r1, const rng::range auto &r2) {
  // std::ranges::views::zip handles this better, but requires range-v3
  for (std::size_t i = 0;
       r1.begin() + i != r1.end() && r2.begin() + i != r2.end(); i++) {
    if (r1[i] != r2[i]) {
      return false;
    }
  }

  return true;
}
