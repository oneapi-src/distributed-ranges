// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"

using T = float;

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;

std::size_t nc = 0;
std::size_t nr = 0;
std::size_t steps = 0;

//
//  debug and verification functions
//

template <std::integral T> bool is_equal(T a, T b) { return a == b; }

template <std::floating_point Tp>
bool is_equal(Tp a, Tp b,
              Tp epsilon = 128 * std::numeric_limits<Tp>::epsilon()) {
  if (a == b) {
    return true;
  }
  auto abs_th = std::numeric_limits<Tp>::min();
  auto diff = std::abs(a - b);
  auto norm =
      std::min(std::abs(a) + std::abs(b), std::numeric_limits<Tp>::max());

  return diff < std::max(abs_th, epsilon * norm);
}

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

auto stencil_op = [](auto &p) {
/* Two notations possible, performance to be verified */
#if 1
  return calculate(p[{-1, 0}], p[{0, 0}], p[{+1, 0}], p[{0, -1}], p[{0, +1}]);
#else
  return calculate(p[-1][0], p[0][0], p[+1][0], p[0][-1], p[0][+1]);
#endif
};

int stencil() {
  dr::halo_bounds hb(1); // 1 row
  dr::mhp::distributed_dense_matrix<T> a(nr, nc, -1, hb), b(nr, nc, -1, hb);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local())
      rng::iota(*r, 10);
  }

  dr::mhp::for_each(b.rows(), [](auto row) { rng::iota(row, 10); });

  // rectgangular subrange of 2d matrix
  auto in = dr::mhp::subrange(a, {1, a.shape()[0] - 1}, {1, a.shape()[1] - 1});
  auto out = dr::mhp::subrange(b, {1, b.shape()[0] - 1}, {1, b.shape()[1] - 1});

  // transform of the above subrange
  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    dr::mhp::transform(in, out.begin(), stencil_op);
    std::swap(in, out);
  }

  dr::mhp::fence();
  MPI_Barrier(MPI_COMM_WORLD);

  if (comm_rank == 0) {
    if (0 == stencil_check(a, b)) {
      fmt::print("stencil check OK!\n");
    } else {
      fmt::print("stencil check failed\n");
      return 1;
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  dr::mhp::init();

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

  nr = options["rows"].as<std::size_t>();
  nc = options["cols"].as<std::size_t>();
  steps = options["steps"].as<std::size_t>();
  std::ofstream *logfile = nullptr;
  if (options.count("log")) {
    logfile = new std::ofstream(fmt::format("dr.{}.log", comm_rank));
    dr::drlog.set_file(*logfile);
  }
  dr::drlog.debug("Rank: {}\n", comm_rank);

  auto error = stencil();
  MPI_Finalize();
  return error;
}
