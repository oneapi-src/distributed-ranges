#include <vector>

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

std::size_t n = 10;

MPI_Comm comm;
int comm_rank;
int comm_size;
const int root_rank = 0;

std::vector<int> a_ref(n);
std::vector<int> b_ref(n);
std::vector<int> full_c_ref(n);
std::vector<int> partial_c_ref(n);

int errors = 0;

void init(auto &a, auto &b, auto &c) {
  if (comm_rank == root_rank) {
    rng::iota(a, 100);
    rng::iota(b, 1000);
  }
  a.fence();
  b.fence();
  c.fence();
}

void show(std::string title, auto &a, auto &b, auto &c, auto &c_ref) {
  if (comm_rank == root_rank) {
    fmt::print("{}:\n    a: {}\n    b: {}\n    c: {}\n  ref: {}\n", title, a, b,
               c, c_ref);
  }
}

void check(std::string title, auto &a, auto &b, auto &c, auto &c_ref) {
  show(title, a, b, c, c_ref);
  if (comm_rank == root_rank) {
    if (rng::equal(c, c_ref)) {
      fmt::print("  Pass\n");
    } else {
      fmt::print("  Fail\n");
      errors++;
    }
  }
}

void vector_add_1() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  if (comm_rank == root_rank) {
    for (std::size_t i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
    }
  }
  c.fence();

  check("vector add 1", a, b, c, full_c_ref);
}

void vector_add_2() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  auto &local_a = a.local();
  auto &local_b = b.local();
  auto &local_c = c.local();

  for (std::size_t i = 0; i < local_a.size(); i++) {
    local_c[i] = local_a[i] + local_b[i];
  }
  c.fence();

  check("vector add 2", a, b, c, full_c_ref);
}

void vector_add_3() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  if (comm_rank == root_rank) {
    rng::transform(a, b, c.begin(), std::plus());
  }
  c.fence();

  check("vector add 3", a, b, c, full_c_ref);
}

void vector_add_4() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  auto &local_a = a.local();
  auto &local_b = b.local();
  auto &local_c = c.local();

  rng::transform(local_a, local_b, local_c.begin(), std::plus());
  c.fence();

  check("vector add 4", a, b, c, full_c_ref);
}

void vector_add_5() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  lib::transform(a, b, c.begin(), std::plus());

  check("vector add 5", a, b, c, full_c_ref);
}

void vector_add_6() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  if (comm_rank == root_rank) {
    rng::transform(a | rng::views::take(4), b, c.begin(), std::plus());
  }

  check("vector add 6", a, b, c, partial_c_ref);
}

void vector_add_7() {
  lib::distributed_vector<int> a(n), b(n), c(n);
  init(a, b, c);

  lib::transform(a | rng::views::take(4), b, c.begin(), std::plus());

  check("vector add 7", a, b, c, partial_c_ref);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  rng::iota(a_ref, 100);
  rng::iota(b_ref, 1000);
  rng::transform(a_ref, b_ref, full_c_ref.begin(), std::plus());
  rng::transform(a_ref | rng::views::take(4), b_ref, partial_c_ref.begin(),
                 std::plus());

  vector_add_1();
  vector_add_2();
  vector_add_3();
  vector_add_4();
  vector_add_6();
  vector_add_7();

  MPI_Finalize();
  return errors;
}
