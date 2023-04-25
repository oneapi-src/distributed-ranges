// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"
#include "dr/mhp/containers/distributed_dense_matrix.hpp"

using T = double;

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

auto stencil_op_verify(std::vector<std::vector<T>> &va,
                       std::vector<std::vector<T>> &vb) {
  for (std::size_t i = 1; i < va.size() - 1; i++) {
    for (std::size_t j = 1; j < va[i].size() - 1; j++)
      vb[i][j] =
          va[i - 1][j] + va[i][j] + va[i + 1][j] + va[i][j - 1] + va[i][j + 1];
  }
}

int rowcmp(dr::mhp::dm_row<T>::iterator r, std::vector<T> &v,
           std::size_t size) {
  int res = 0;
  for (std::size_t _i = 0; _i < size; _i++) {
    if (r[_i] != v[_i]) {
      fmt::print("{}: Fail (r[{}] = {}, v[{}] = {})\n", comm_rank, _i, r[_i],
                 _i, v[_i]);
      res = -1;
    }
  }
  return res;
}

int local_compare(std::string label, dr::mhp::distributed_dense_matrix<T> &dm,
                  std::vector<std::vector<T>> &vv) {
  int res = 0;
  for (auto r = dm.rows().begin(); r != dm.rows().end(); r++) {
    if (r.is_local())
      if (-1 == rowcmp((*r).begin(), vv[(*r).idx()], (*r).size())) {
        fmt::print("{}: {} Fail (idx = {})\n", comm_rank, label, (*r).idx());
        res = -1;
      }
  }
  return res;
}

int check(dr::mhp::distributed_dense_matrix<T> &a,
          dr::mhp::distributed_dense_matrix<T> &b) {

  std::vector<std::vector<T>> va(nr), vb(nr);

  for (auto r = va.begin(); r != va.end(); r++) {
    (*r).resize(nc);
    std::iota((*r).begin(), (*r).end(), 10);
  }

  for (auto r = vb.begin(); r != vb.end(); r++) {
    (*r).resize(nc);
    std::iota((*r).begin(), (*r).end(), 10);
  }

  for (std::size_t s = 0; s < steps; s++) {
    stencil_op_verify(va, vb);
    std::swap(va, vb);
  }

  return local_compare("A", a, (steps % 2) ? vb : va) +
         local_compare("B", b, (steps % 2) ? va : vb);
}

//
// stencil 1 - operation on subsequent cells
//

auto stencil_op1 = [](auto &p) {
  // Two notations possible, performance to be verified
  // T res = p[{-1, 0}] + p[{0, 0}] + p[{+1, 0}] + p[{0, -1}] + p[{0, +1}];
  // the version below (buggy for low egde of the matrix - to be fixed)
  T res = p[-1][0] + p[0][0] + p[+1][0] + p[0][-1] + p[0][+1];

  return res;
};

int stencil1() {
  dr::halo_bounds hb(1); // 1 row
  dr::mhp::distributed_dense_matrix<T> a(nr, nc, -1, hb), b(nr, nc, -1, hb);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local()) {
      std::iota((*r).begin(), (*r).end(), 10);
    }
  }

  // the same operation on each row
  dr::mhp::for_each(b.rows(),
                    [](auto &row) { std::iota(row.begin(), row.end(), 10); });

  // rectgangular subrange of 2d matrix
  auto in = dr::mhp::subrange(a, {1, a.shape()[0] - 1}, {1, a.shape()[1] - 1});
  auto out = dr::mhp::subrange(b, {1, b.shape()[0] - 1}, {1, b.shape()[1] - 1});

  // transform of the above subrange
  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    dr::mhp::transform(in, out.begin(), stencil_op1);
    std::swap(in, out);
  }

  if (0 == check(a, b))
    fmt::print("{}: stencil1 check OK!\n", comm_rank);
  else
    fmt::print("{}: stencil1 check failed\n", comm_rank);

  return 0;
}

//
// stencil 2 - operation on subsequent rows
//

auto stencil_op2 = [](auto &&p) {
  dr::mhp::dm_row<T> out_row((*p).size());
  out_row[0] = p[0][0];
  for (std::size_t i = 1; i < nc - 1; i++) {
    out_row[i] = p[-1][i] + p[0][i - 1] + p[0][i] + p[0][i + 1] + p[1][i];
  }
  out_row[nc - 1] = p[0][nc - 1];

  return out_row;
};

int stencil2() {
  dr::halo_bounds hb(1); // 1 row
  dr::mhp::distributed_dense_matrix<T> a(nr, nc, hb), b(nr, nc, hb);

  dr::mhp::for_each(a.rows(),
                    [](auto &row) { std::iota(row.begin(), row.end(), 10); });
  dr::mhp::for_each(b.rows(),
                    [](auto &row) { std::iota(row.begin(), row.end(), 10); });

  // all rows except 1st and last

  auto in = rng::subrange(a.rows().begin() + 1, a.rows().end() - 1);
  auto out = rng::subrange(b.rows().begin() + 1, b.rows().end() - 1);

  for (std::size_t s = 0; s < steps; s++) {
    dr::mhp::halo(in).exchange();
    // rng::for_each(rng::views::zip(in, out), stencil_op2); // to consider
    dr::mhp::transform(in, out.begin(), stencil_op2);
    std::swap(in, out);
  }

  if (0 == check(a, b))
    fmt::print("{}: stencil2 check OK!\n", comm_rank);
  else
    fmt::print("{}: stencil2 failed\n", comm_rank);

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

  auto error1 = stencil1();
  auto error2 = stencil2();
  MPI_Finalize();
  return error1 + error2;
}
