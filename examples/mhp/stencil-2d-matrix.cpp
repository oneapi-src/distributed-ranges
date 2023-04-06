// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"
#include "dr/mhp/containers/distributed_dense_matrix.hpp"
#include "dr/mhp/containers/subrange.hpp"

using T = float;

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;

std::size_t m = 0;
std::size_t n = 0;
std::size_t steps = 0;

auto stencil_op = [](auto &p) {
  T res = p[{-1, 0}] + p[{0, 0}] + p[{+1, 0}] + p[{0, -1}] + p[{0, +1}];
  return res;
};

auto stencil_op_v (std::vector<std::vector<T>> & va, std::vector<std::vector<T>> & vb) {

  for (std::size_t i = 1; i < va.size() - 1; i++)
    for (std::size_t j = 1; j < va[i].size() - 1; j++)
      vb[i][j] = va[i-1][j] + va[i][j] + va[i+1][j] + va[i][j-1] + va[i][j+1];

}

int stencil() {
  lib::halo_bounds hb(1); // 1 row
  mhp::distributed_dense_matrix<T> a(n, m, -1, hb), b(n, m, -1, hb);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local())
      std::iota((*r).begin(), (*r).end(), (*r).idx() * 10);
  }

  // the same operation on each row
  mhp::for_each(b.rows(),
                [](auto &row) { std::fill(row.begin(), row.end(), 0); });


  auto in = mhp::dm_subrange(a, {1, a.shape()[0] - 1}, {1, a.shape()[1] - 1});
  auto out = mhp::dm_subrange(b, {1, b.shape()[0] - 1}, {1, b.shape()[1] - 1});

  for (std::size_t s = 0; s < steps; s++) {
    mhp::halo(in).exchange();
    mhp::dm_transform(in, out.begin(), stencil_op);
    std::swap(in, out);
  }

  // if (comm_rank == 0) {
  //   return check(in);
  // }

  a.dump_matrix("final a");
  b.dump_matrix("final b");

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
    ("log", "Enable logging")
    ("rows", "Number of rows", cxxopts::value<std::size_t>()->default_value("20"))
    ("cols", "Number of columns", cxxopts::value<std::size_t>()->default_value("10"))
    ("steps", "Number of time steps", cxxopts::value<std::size_t>()->default_value("3"))
    ("help", "Print help");
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  n = options["rows"].as<std::size_t>();
  m = options["cols"].as<std::size_t>();
  steps = options["steps"].as<std::size_t>();
  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    lib::drlog.set_file(*logfile);
  }
  lib::drlog.debug("Rank: {}\n", comm_rank);

  auto error = stencil();
  MPI_Finalize();
  return error;
}
