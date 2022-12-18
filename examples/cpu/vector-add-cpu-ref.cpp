#include "mpi.h"

#include "utils.hpp"
#include "vector-add-serial.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

void vector_add() {
  using T = int;

  const size_t slice_size = 5;
  const size_t full_size = slice_size * comm_size;

  // Compute the reference data
  vector_add_serial<T> ref_adder;
  if (comm_rank == 0) {
    ref_adder.init(full_size);
    ref_adder.compute();
  }

  // Distribute the data
  auto data_type = mpi_data_type<T>();
  std::vector<T> a(slice_size), b(slice_size), c(slice_size);
  MPI_Scatter(ref_adder.a.data(), slice_size, data_type, a.data(), slice_size,
              data_type, 0, comm);
  MPI_Scatter(ref_adder.b.data(), slice_size, data_type, b.data(), slice_size,
              data_type, 0, comm);

// multi-threaded vector add on slice
#pragma omp parallel for
  for (size_t i = 0; i < slice_size; i++) {
    c[i] = a[i] + b[i];
  }

  // Collect the results
  std::vector<T> c_full(full_size);
  MPI_Gather(c.data(), slice_size, data_type, c_full.data(), slice_size,
             data_type, 0, comm);

  // Check
  if (comm_rank == 0) {
    fmt::print("a: {}\n", ref_adder.a);
    fmt::print("b: {}\n", ref_adder.b);
    fmt::print("c: {}\n", c_full);
    ref_adder.check(c_full);
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
