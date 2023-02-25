// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/mhp.hpp"

using T = int;

MPI_Comm comm;
int comm_rank;

const std::size_t n = 10;

void vector_add() {
  mhp::distributed_vector<T> a(n), b(n), c(n);

  // Initialize
  mhp::iota(a, 10);
  mhp::iota(b, 100);

  auto add = [](auto ops) { return ops.first + ops.second; };

  mhp::transform(rng::views::zip(a, b), c.begin(), add);

  if (comm_rank == 0) {
    fmt::print("a: {}\n"
               "b: {}\n"
               "c: {}\n",
               a, b, c);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  mhp::init();

  vector_add();

  MPI_Finalize();
  return 0;
}
