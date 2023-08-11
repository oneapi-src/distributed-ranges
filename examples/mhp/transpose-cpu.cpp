// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

#include "transpose-serial.hpp"
#include "utils.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;
const int root_rank = 0;

bool is_root() { return root_rank == comm_rank; }
void transpose() {
  using T = double;

  const std::size_t m_segment = 2;
  // segment has entire row, but it must divide evenly because of the transpose
  const std::size_t n_segment = 6;
  const std::size_t m = m_segment * comm_size;
  const std::size_t n = n_segment * comm_size;

  dr::distributed_vector<T, dr::block_cyclic> dv_a(n), dv_b(n);
  dr::distributed_mdspan<T, std::dextents<2>> dm_aT(dv_b.data(), m, n);

  // root initializes dv_a
  transpose_serial<T> ref_transpose;
  if (is_root()) {
    ref_transpose.init(m, n);
    ref_transpose.compute();
    std::copy(ref_transpose.a.begin(), ref_transpose.a.end(), dv_a);
  }

  // Transpose my segment
  std::vector<T> local(m_segment * n);
  stdex::mdarray<T, stdex::dextents<std::size_t, 2>> lm(
  transpose(m_segment, n, dv_a.local_segment().data(), local.data());

  // Copy my segment to target blocks
  auto block_size = m_segment * n_segment;
  for (int r = 0; r < comm_size; r++) {
    // Create a view for the remote target block
    // We have block (src_rank, target_rank), store at (target_rank, src_rank)
    auto target_block = dr::distributed_submdspan(
        dm, std::vector({std::pair(r * m_segment, (r + 1) * m_segment),
                         std::pair(comm_rank * n_segment,
                                   (comm_rank + 1) * n_segment)}));

    std::copy(local.data() + r * block_size,
              local.data() + (r + 1) * block_size, target_block);
  }

  std::vector<T> result(m * n);
  std::copy(dv_b.begin(), dv_b.end(), result.begin());
  ref_transpose.check(result);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  transpose();

  dr::mhp::finalize();
  MPI_Finalize();
  return 0;
}
