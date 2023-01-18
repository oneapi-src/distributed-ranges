// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-fuzz.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  MPI_Init(argc, argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  return 0;
}

enum class test_type { algorithm };

struct fuzz_spec {
  int algorithm;
  std::size_t n;
  std::size_t b;
  std::size_t e;
};

extern "C" int LLVMFuzzerTestOneInput(const fuzz_spec *fuzz, size_t size) {
  if (sizeof(fuzz_spec) < size)
    return 0;

  auto n = fuzz->n;
  auto b = fuzz->b;
  auto e = fuzz->e;
  if (n > 64 || b > n || e > n || b > e || n == 0)
    return 0;

  // fmt::print("n: {} b: {} e: {}\n", n, b, e);

  switch (fuzz->algorithm) {
  case 0:
    // check_copy(n, b, e);
    break;
  case 1:
    check_transform(n, b, e);
    break;
  default:
    break;
  }

  return 0;
}
