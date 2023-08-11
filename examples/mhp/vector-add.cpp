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
  dr::mhp::distributed_vector<T> a(n), b(n), c(n);

  // Initialize
  dr::mhp::iota(a, 10);
  dr::mhp::iota(b, 100);

  auto add = [](auto ops) { return ops.first + ops.second; };

  dr::mhp::transform(dr::mhp::views::zip(a, b), c.begin(), add);

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
  dr::mhp::init();

  vector_add();

  dr::mhp::finalize();
  MPI_Finalize();
  return 0;
}
