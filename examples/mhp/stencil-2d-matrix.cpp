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


constexpr size_t Size = 10;
using ET = int;

cxxopts::ParseResult options;

void dump_matrix( std::string msg, mhp::distributed_dense_matrix<ET> & dm) {
  std::stringstream s;
  s << comm_rank << ": " << msg << " :\n";
  for( auto r: dm.local_rows()) {
    s << comm_rank << ": row : ";
    for (auto _i = r.begin(); _i != r.end(); ++_i) 
      s << *_i << " ";
    s << ENDL;
  }
  std::cout << s.str();
}


int stencil(const size_t n, auto steps) {
  // lib::halo_bounds hb(1);
  mhp::distributed_dense_matrix<ET> a(n, n);

  // segfaults with mpirun
  //
  // auto _rows = a.rows();
  // for_each(_rows.begin(), _rows.end(), [](auto &&row) { std::iota(row.begin(), row.end(), Size); });
  // dump_matrix("After for_each", a);

  int init = 0;
  for( auto r: a.local_rows()) {
    std::iota(r.begin(), r.end(), (init++) * Size + mhp::default_comm().rank());
  }
  dump_matrix("After for", a);

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

// return locally stored elements as a vector
auto local_vector(auto &&dr) {
  auto lvector = rng::views::zip( local_segments(dr) );
  return std::get<0>( *lvector.begin() );
}
