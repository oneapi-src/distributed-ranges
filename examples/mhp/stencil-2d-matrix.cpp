// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"
#include "dr/mhp/containers/distributed_dense_matrix.hpp"

MPI_Comm comm;
int comm_rank;
int comm_size;

constexpr std::size_t Size = 10;
using eltype = int;

cxxopts::ParseResult options;

void dump_matrix(std::string msg, mhp::distributed_dense_matrix<eltype> &dm) {
  std::stringstream s;
  s << comm_rank << ": " << msg << " :\n";
  for (auto r : dm.local_rows()) {
    s << comm_rank << ": row " << r.idx() << " : ";
    for (auto _i = r.begin(); _i != r.end(); ++_i)
      s << *_i << " ";
    s << std::endl;
  }
  std::cout << s.str();
}

void dump_vector(std::string msg, mhp::distributed_vector<eltype> &v) {
  std::stringstream s;
  s << comm_rank << ": " << msg << " :\n";
  mhp::for_each(v, [&s](auto &&el) { s << comm_rank << ":" << el << std::endl; });
  std::cout << s.str();
}

int stencil(const std::size_t n, auto steps) {
  lib::halo_bounds hb(1); // number of rows
  mhp::distributed_dense_matrix<eltype> a(n, n, hb);

  for (mhp::dm_row<eltype> r : a.local_rows()) {
    std::fill(r.begin(), r.end(), -1);
  }
  dump_matrix("Filled with -1", a);

  for (mhp::dm_row<eltype> r : a.local_rows()) {
    std::fill(r.begin() + 1, r.end() - 1,
              Size * r.idx() + mhp::default_comm().rank());
  }
  dump_matrix("After iteration over local_rows()", a);

  // mhp::for_each(a.rows(), [](mhp::dm_row<eltype> &row) {
  //   std::iota(row.begin(), row.end(), Size * row.idx());
  // });
  // dump_matrix("After for_each", a);

  for (mhp::dm_row<eltype> r : a.local_rows()) {
    std::fill(r.begin(), r.end(), Size * r.idx() + mhp::default_comm().rank());
  }

  dump_matrix("DM before exchange", a);

  // auto mxrng = rng::subrange(a.rows().begin(), a.rows().end());
  auto mxrng = rng::subrange(a.begin(), a.end());

  mhp::halo(mxrng).exchange();
  dump_matrix("DM After exchange", a);

  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  mhp::init();

  cxxopts::Options options_spec(argv[0], "stencil 2d");
  // clang-format off
  options_spec.add_options()
    ("n", "Size n of array", cxxopts::value<std::size_t>()->default_value(std::to_string(Size)))
    ("s", "Number of time steps", cxxopts::value<std::size_t>()->default_value("5"))
    ("help", "Print help");
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  auto error =
      stencil(options["n"].as<std::size_t>(), options["s"].as<std::size_t>());

  MPI_Finalize();
  return error;
}
