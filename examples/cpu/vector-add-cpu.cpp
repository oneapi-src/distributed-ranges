#include "mpi.h"

#include "dr/distributed-ranges.hpp"

#include "utils.hpp"
#include "vector-add-serial.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;
const int root_rank = 0;

bool is_root() { return root_rank == comm_rank; }

void vector_add() {
  using T = int;

  // size of distributed vector
  const std::size_t segment_size = 5;
  const size_t n = segment_size * comm_size;

  // Compute the reference data
  vector_add_serial<T> ref_adder;
  if (is_root()) {
    ref_adder.init(n);
    ref_adder.compute();
  }

  lib::distributed_vector<T> dv_a(n), dv_b(n), dv_c(n);

  // Distribute the data
  lib::collective::copy(root_rank, ref_adder.a, dv_a);
  lib::collective::copy(root_rank, ref_adder.b, dv_b);

  // c = a + b
  lib::transform(dv_a, dv_b, dv_c.begin(), std::plus<>());

  // Collect the results
  std::vector<T> result(n);
  lib::collective::copy(root_rank, dv_c, result);

  // Check
  if (is_root()) {
    show("a: ", ref_adder.a);
    show("b: ", ref_adder.b);
    show("c: ", result);
    ref_adder.check(result);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  vector_add();

  MPI_Finalize();
  return 0;
}
