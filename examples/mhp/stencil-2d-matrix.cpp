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

std::size_t m = 10;
std::size_t n = 10;
std::size_t steps = 0;

auto stencil_op1 = [](auto &&v) {
  auto &[in_row, out_row] = v;
  auto p = &in_row;
  for (std::size_t i = 1; i < m - 1; i++) {
    out_row[i] = p[-1][i] + p[0][i - 1] + p[0][i] + p[0][i + 1] + p[1][i];
  }
};

auto stencil_op2 = [](auto &&v) {
  auto p = &v;
  return p[-1] + p[0] + p[+1] + p[-n] + p[+n];
};
/*
auto format_matrix(auto &&m) {
  std::string str;
  for (auto &&row : m) {
    str += fmt::format("  {}\n", Row(row));
  }
  return str;
}

auto equal(auto &&a, auto &&b) {
  for (std::size_t i = 0; i < a.size(); i++) {
    if (Row(a[i]) != Row(b[i])) {
      return false;
    }
  }
  return true;
}

auto compare(auto &&ref, auto &&actual) {
  if (equal(ref, actual)) {
    return 0;
  }

  fmt::print("Mismatch\n");
  if (rows <= 10 && cols <= 10) {
    fmt::print("ref:\n{}\nactual:\n{}\n", format_matrix(ref),
               format_matrix(actual));
  }

  return 1;
}

int check(auto &&actual) {
  // Serial stencil
  std::vector<Row> a(rows), b(rows);
  rng::for_each(a, [](auto &row) { rng::iota(row, 100); });
  rng::for_each(b, [](auto &row) { rng::fill(row, 0); });

  auto in = rng::subrange(a.begin() + 1, a.end() - 1);
  auto out = rng::subrange(b.begin() + 1, b.end() - 1);
  for (std::size_t s = 0; s < steps; s++) {
    rng::for_each(rng::views::zip(in, out), stencil_op);
    std::swap(in, out);
  }

  // Check the result
  return compare(steps % 2 ? b : a, actual);
}
 */
// int stencil_1() {
//   lib::halo_bounds hb(1); // 1 row
//   mhp::distributed_dense_matrix<T> a(n, m, hb), b(n, m, hb);

//   mhp::for_each(a.rows(),
//                 [](auto &row) { std::iota(row.begin(), row.end(), 100); });
//   mhp::for_each(b.rows(),
//                 [](auto &row) { std::fill(row.begin(), row.end(), 0); });

//   // all rows except 1st and last
//   auto in = rng::subrange(a.rows().begin() + 1, a.rows().end() - 1);
//   auto out = rng::subrange(b.rows().begin() + 1, b.rows().end() - 1);

//   for (std::size_t s = 0; s < steps; s++) {
//     mhp::halo(in).exchange();
//     mhp::for_each(rng::views::zip(in, out), stencil_op1);
//     std::swap(in, out);
//   }

//   /* auto error = 0;
//   if (comm_rank == 0) {
//     error = check(steps % 2 ? b : a);

//     if (error) {
//       fmt::print("Fail\n");
//     } else {
//       fmt::print("Pass\n");
//     }
//   } */

//   MPI_Barrier(comm);
//   return error;
// }

int stencil_2() {
  lib::halo_bounds hb(1); // 1 row
  mhp::distributed_dense_matrix<T> a(n, m, hb), b(n, m, hb);

  mhp::for_each(a.rows(),
                [](auto &row) { std::iota(row.begin(), row.end(), 100); });
  mhp::for_each(b.rows(),
                [](auto &row) { std::fill(row.begin(), row.end(), 0); });

  // to implement
  auto in = mhp::subrange(a, {1, 1}, {a.shape()[0] - 1, a.shape()[1] - 1});
  auto out = mhp::subrange(b, {1, 1}, {a.shape()[0] - 1, a.shape()[1] - 1});

  int _i = lib::ranges::segments(in);
  // rng::begin(lib::ranges::segments(in)[0]); //.halo();

  // for (std::size_t s = 0; s < steps; s++) {
  //   mhp::halo(in).exchange();
  //   mhp::transform(in, out.begin(), stencil_op2);
  //   std::swap(in, out);
  // }

  /* if (comm_rank == 0) {
    return check(in, n, steps);
  } else {
    return 0;
  } */
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
    ("rows", "Number of rows", cxxopts::value<std::size_t>()->default_value("10"))
    ("cols", "Number of columns", cxxopts::value<std::size_t>()->default_value("10"))
    ("steps", "Number of time steps", cxxopts::value<std::size_t>()->default_value("5"))
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

  // version _1 or _2, of choice
  // auto error = stencil_1();
  auto error = stencil_2();
  MPI_Finalize();
  return error;
}
