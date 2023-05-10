// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"

using T = float;

MPI_Comm comm;
int comm_rank;
int comm_size;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  dr::mhp::init();

  std::size_t nc = 16;
  std::size_t nr = 16;

  {

    dr::halo_bounds hb(1); // 1 row
    dr::mhp::distributed_dense_matrix<T> a(nr, nc, -1, hb);

    // different operation on every row - user must be aware of rows
    // distribution
    for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
      if (r.is_local())
        rng::iota(*r, 10);
    }

    for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
      if (r.is_local()) {
        auto &&row = *r;
        fmt::print("{}\n", row);
      }
    }

    dr::mhp::fence();
    fflush(stdout);
    MPI_Barrier(comm);
    fflush(stdout);
    MPI_Barrier(comm);

    if (comm_rank == 0) {

      fmt::print("Printing out whole matrix:\n");
      for (auto &&tile : a.tile_segments()) {
        for (auto &&[index, value] : tile) {
          auto &&[i, j] = index;
          fmt::print("({}, {}): {}\n", i, j, value);
        }
      }

      fmt::print("Printing out individual tiles:\n");
      for (std::size_t i = 0; i < a.grid_shape()[0]; i++) {
        for (std::size_t j = 0; j < a.grid_shape()[1]; j++) {
          fmt::print("Tile {}, {}\n", i, j);

          auto &&tile = a.tile({i, j});

          for (auto &&[index, value] : tile) {
            auto &&[i, j] = index;
            fmt::print("({}, {}): {}\n", i, j, value);
          }
        }
      }
    }

    MPI_Barrier(comm);
  }

  MPI_Finalize();
  return 0;
}
