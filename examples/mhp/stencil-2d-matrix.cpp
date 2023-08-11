// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../include/data-utils.hpp"

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include "mpi.h"

#include <algorithm>

using T = float;

int comm_rank;
int comm_size;

cxxopts::ParseResult options;

std::size_t nc;
std::size_t nr;
std::size_t steps;
std::size_t stencil_choice;

//
//  debug and verification functions
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

auto stencil_op_verify(std::vector<std::vector<T>> &va,
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

  for (auto r = va.begin(); r != va.end(); r++) {
    (*r).resize(nc);
    rng::iota(*r, 10);
  }

  for (auto r = vb.begin(); r != vb.end(); r++) {
    (*r).resize(nc);
    rng::iota(*r, 10);
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

/* first access a row, then element in row */
auto stencil_op_two_step_access = [](auto &p) {
  return calculate(p[-1][0], p[0][0], p[+1][0], p[0][-1], p[0][+1]);
};

/* access element at once, with 2 coordinates */
auto stencil_op_one_step_access = [](auto &p) {
  return calculate(p[{-1, 0}], p[{0, 0}], p[{+1, 0}], p[{0, -1}], p[{0, +1}]);
};

int stencil(auto stencil_op) {
  auto dist = dr::mhp::distribution().halo(1, 1); // 1 row
  dr::mhp::distributed_dense_matrix<T> a(nr, nc, -1, dist), b(nr, nc, -1, dist);

  // 1st approach - different operation for each row is possible here
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local())
      rng::iota(*r, 10);
  }

  // 2nd approach - the same operation for each row
  dr::mhp::for_each(b.rows(), [](auto row) { rng::iota(row, 10); });

  // rectgangular subrange of 2d matrix
  auto in = dr::mhp::subrange(a, {1, a.shape()[0] - 1}, {1, a.shape()[1] - 1});
  auto out = dr::mhp::subrange(b, {1, b.shape()[0] - 1}, {1, b.shape()[1] - 1});

  // transform of the above subrange
  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    dr::mhp::transform(in, out.begin(), stencil_op);
    std::swap(in, out);
    dr::mhp::barrier();
  }

  dr::mhp::barrier();

  if (comm_rank == 0) {
    if (0 == stencil_check(a, b)) {
      fmt::print("stencil-2d-matrix ({}-step stencil): check OK!\n",
                 stencil_choice);
    } else {
      fmt::print("stencil-2d-matrix ({}-step stencil): check failed\n",
                 stencil_choice);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {

  cxxopts::Options options_spec(argv[0], "stencil-2d-matrix");

  // clang-format off
  options_spec.add_options()
    ("log", "Enable logging")
    ("rows", "Number of rows", cxxopts::value<std::size_t>()->default_value("20"))
    ("cols", "Number of columns", cxxopts::value<std::size_t>()->default_value("10"))
    ("steps", "Number of time steps", cxxopts::value<std::size_t>()->default_value("5"))
    ("stencil", "Choice of stencil, 1-step or 2-step", cxxopts::value<std::size_t>()->default_value("1"))
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
  stencil_choice = options["stencil"].as<std::size_t>();

  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  dr::mhp::init();

  int error = -1;
  switch (stencil_choice) {
  case 1:
    error = stencil(stencil_op_one_step_access);
    break;
  case 2:
    error = stencil(stencil_op_two_step_access);
    break;
  default:
    fmt::print("Error: stencil arg should be 1 for 1-step stencil or 2 for "
               "2-step stencil\n");
    break;
  }

  dr::mhp::finalize();
  MPI_Finalize();
  return error;
}
