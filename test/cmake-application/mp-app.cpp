// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/mp.hpp"

using T = int;

MPI_Comm comm;
int comm_rank;

const std::size_t n = 10;

void vector_add() {
  dr::mp::distributed_vector<T> a(n), b(n), c(n);

  // Initialize
  dr::mp::iota(a, 10);
  dr::mp::iota(b, 100);

  auto add = [](auto ops) { return ops.first + ops.second; };

  dr::mp::transform(dr::mp::views::zip(a, b), c.begin(), add);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  dr::mp::init();

  vector_add();

  MPI_Finalize();
  return 0;
}
