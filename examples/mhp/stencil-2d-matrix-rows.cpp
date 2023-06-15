// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>

#include "../include/data-utils.hpp"
#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"

using T = float;

int comm_rank;

cxxopts::ParseResult options;

std::size_t nc = 0;
std::size_t nr = 0;
std::size_t steps = 0;

//
//  verification
//
int matrix_compare(std::string label, dr::mhp::distributed_dense_matrix<T> &dm,
                   std::vector<std::vector<T>> &vv) {
  int res = 0;
  for (std::size_t i = 0; i < vv.size(); i++) {
    for (std::size_t j = 0; j < vv[i].size(); j++) {
      if (!is_equal(vv[i][j], (T)dm.begin()[{i, j}])) {
        fmt::print("{}: {} Fail in cell [{},{}] values: ref {} matrix {})\n",
                   comm_rank, label, i, j, vv[i][j], (T)dm.begin()[{i, j}]);
        res = -1;
      }
    }
  }
  return res;
}

//
// calculation formula
//
T calculate(T arg1, T arg2, T arg3, T arg4, T arg5) {
  return arg1 + arg2 + arg3 + arg4 + arg5;
}

//
// verification of results
//
void stencil_op_verify(std::vector<std::vector<T>> &va,
                       std::vector<std::vector<T>> &vb) {
  for (std::size_t i = 1; i < va.size() - 1; i++) {
    for (std::size_t j = 1; j < va[i].size() - 1; j++)
      vb[i][j] = calculate(va[i - 1][j], va[i][j], va[i + 1][j], va[i][j - 1],
                           va[i][j + 1]);
  }
}

int stencil_check(dr::mhp::distributed_dense_matrix<T> &a,
                  dr::mhp::distributed_dense_matrix<T> &b) {

  std::vector<std::vector<T>> va(nr), vb(nr);

  for (auto &r : va) {
    r.resize(nc);
    rng::iota(r, 10);
  }

  for (auto &r : vb) {
    r.resize(nc);
    rng::iota(r, 10);
  }

  for (std::size_t s = 0; s < steps; s++) {
    stencil_op_verify(va, vb);
    std::swap(va, vb);
  }

  return matrix_compare("A", a, (steps % 2) ? vb : va) +
         matrix_compare("B", b, (steps % 2) ? va : vb);
}

//
// stencil
//

auto stencil_op = [](auto &&p) {
  std::vector<T> out_row((*p).size());

  out_row[0] = p[0][0];
  for (std::size_t i = 1; i < nc - 1; i++) {
    out_row[i] =
        calculate(p[-1][i], p[0][i - 1], p[0][i], p[0][i + 1], p[1][i]);
  }
  out_row[nc - 1] = p[0][nc - 1];
  return out_row;
};

int stencil() {
  dr::halo_bounds hb(1); // 1 row
  dr::mhp::distributed_dense_matrix<T> a(nr, nc, -1, hb), b(nr, nc, -1, hb);

  // the same operation on every row
  dr::mhp::for_each(a.rows(), [](auto row) { rng::iota(row, 10); });

  // different operation on every row is posiible by access to the row (*r)
  for (auto r = b.rows().begin(); r != b.rows().end(); r++) {
    if (r.is_local())
      rng::iota(*r, 10);
  }

  // subranges of rows
  auto in = rng::subrange(a.rows().begin() + 1, a.rows().end() - 1);
  auto out = rng::subrange(b.rows().begin() + 1, b.rows().end() - 1);

  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    dr::mhp::transform(in, out.begin(), stencil_op);
    std::swap(in, out);
  }

  dr::mhp::fence();
  MPI_Barrier(MPI_COMM_WORLD);

  // verify the result of operation by comparison with sequential
  // version on std structures
  if (comm_rank == 0) {
    if (0 == stencil_check(a, b)) {
      fmt::print("stencil-2d-matrix-rows: check OK!\n");
    } else {
      fmt::print("stencil-2d-matrix-rows: check failed\n");
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {

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

  if (options.count("help")) {
    std::cout << options_spec.help() << std::endl;
    exit(0);
  }

  nr = options["rows"].as<std::size_t>();
  nc = options["cols"].as<std::size_t>();
  steps = options["steps"].as<std::size_t>();
  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  dr::mhp::init();

  auto error = stencil();

  MPI_Finalize();
  return error;
}
