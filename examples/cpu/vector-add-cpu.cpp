#include "mpi.h"

#include "utils.hpp"
#include "vector-add-serial.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

void vector_add() {
  using T = int;

  // size of distributed vector
  const size_t n = 5 * comm_size;

  // Compute the reference data
  vector_add_serial<T> ref_adder;
  if (comm_rank == 0) {
    ref_adder.init(n);
    ref_adder.compute();
  }

  // Default to block distribution
  // communicator goes here?
  auto dist = lib::block_cyclic(2, comm);
  lib::distributed_vector<T> a(n, dist), b(n, dist), c(n, dist);

  // Distribute the data
  a = ref_adder.a;
  b = ref_adder.b;

  // This is ok for 1 rank/core.
  // What if I want to use 1 rank/node and openmp within the node?
  // Need foreach slice and then openmp loop nested inside.
  //
  // #pragma omp parallel for
  lib::for_each(lib::parallel_explicit(), std::ranges::iota_view<size_t>(0, n),
                [&](size_t i) { c[i] = a[i] + b[i]; });

  // Collect the results
  std::vector<T> result(n);
  // No way to say that I only need the result on root?
  result = c;

  // Check
  if (comm_rank == 0) {
    show("a: ", ref_adder.a);
    show("b: ", ref_adder.b);
    show("c: ", c_full);
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
