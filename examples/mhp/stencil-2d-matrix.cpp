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

std::size_t nc = 0;
std::size_t nr = 0;
std::size_t steps = 0;

void dump_v(std::string msg, std::vector<std::vector<T>> &vv) {
  std::stringstream s;
  int idx = 0;
  s << comm_rank << ": " << msg << std::endl;
  for (auto r : vv) {
    s << comm_rank << ": "
      << "row : " << idx++;
    for (auto el : r)
      s << " " << el;
    s << std::endl;
  }
  std::cout << s.str();
}

auto stencil_op = [](auto &p) {
  T res = p[{-1, 0}] + p[{0, 0}] + p[{+1, 0}] + p[{0, -1}] + p[{0, +1}];

  // the version below is intended to work, too, still a bug is present when
  // refering to negative index reaching halo area T res = p[-1][0] + p[0][0] +
  // p[+1][0] + p[0][-1] + p[0][+1];

  return res;
};

auto stencil_op_v(std::vector<std::vector<T>> &va,
                  std::vector<std::vector<T>> &vb) {
  for (std::size_t i = 1; i < va.size() - 1; i++) {
    for (std::size_t j = 1; j < va[i].size() - 1; j++)
      vb[i][j] =
          va[i - 1][j] + va[i][j] + va[i + 1][j] + va[i][j - 1] + va[i][j + 1];
  }
}

int rowcmp(mhp::dm_row<T>::iterator r, std::vector<T> &v, std::size_t size) {
  for (std::size_t _i = 0; _i < size; _i++) {
    if (r[_i] != v[_i]) {
      fmt::print("{}: Fail (r[{}] = {}, v[{}] = {})\n", comm_rank, _i, r[_i],
                 _i, v[_i]);
      // return -1;
    }
  }
  return 0;
}

int local_compare(mhp::distributed_dense_matrix<T> &dm,
                  std::vector<std::vector<T>> &vv) {

  for (auto r = dm.rows().begin(); r != dm.rows().end(); r++) {
    if (r.is_local())
      if (-1 == rowcmp((*r).begin(), vv[(*r).idx()], (*r).size())) {
        fmt::print("{}: Fail (idx = {})", comm_rank, (*r).idx());
        return -1;
      }
  }
  return 0;
}

int check(mhp::distributed_dense_matrix<T> &a,
          mhp::distributed_dense_matrix<T> &b) {
  std::vector<std::vector<T>> va(nr), vb(nr);
  for (auto r = va.begin(); r != va.end(); r++) {
    r->resize(nc);
    std::iota(r->begin(), r->end(), 10);
  }

  for (auto r = vb.begin(); r != vb.end(); r++) {
    r->resize(nc);
    std::iota(r->begin(), r->end(), 10);
  }

  for (std::size_t s = 0; s < steps; s++) {
    stencil_op_v(va, vb);
    std::swap(va, vb);
  }
  // if (0 == comm_rank) dump_v("Final va", va);
  // if (0 == comm_rank) dump_v("Final vb", vb);
  return local_compare(a, (steps % 2) ? vb : va);
}

int stencil() {
  lib::halo_bounds hb(1); // 1 row
  mhp::distributed_dense_matrix<T> a(nr, nc, -1, hb), b(nr, nc, -1, hb);

  // different operation on every row - user must be aware of rows distribution
  for (auto r = a.rows().begin(); r != a.rows().end(); r++) {
    if (r.is_local())
      std::iota((*r).begin(), (*r).end(), 10);
  }

  // the same operation on each row
  mhp::for_each(b.rows(),
                [](auto &row) { std::iota(row.begin(), row.end(), 10); });

  auto in = mhp::subrange(a, {1, a.shape()[0] - 1}, {1, a.shape()[1] - 1});
  auto out = mhp::subrange(b, {1, b.shape()[0] - 1}, {1, b.shape()[1] - 1});

  for (std::size_t s = 0; s < steps; s++) {
    mhp::halo(in).exchange();
    mhp::transform(in, out.begin(), stencil_op);
    std::swap(in, out);
  }

  // a.dump_matrix("final a");
  // b.dump_matrix("final b");

  check(a, b);

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

  nr = options["rows"].as<std::size_t>();
  nc = options["cols"].as<std::size_t>();
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
