// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "mpi.h"

#include "dr/mhp.hpp"

using T = float;

MPI_Comm comm;
int comm_rank;
int comm_size;

cxxopts::ParseResult options;

const std::size_t cols = 10;
std::size_t rows = 10;
std::size_t steps = 0;

using Row = std::array<T, cols>;

auto stencil_op = [](auto &&v) {
  auto &[in_row, out_row] = v;
  auto p = &in_row;
  for (std::size_t i = 1; i < cols - 1; i++) {
    out_row[i] = p[-1][i] + p[0][i - 1] + p[0][i] + p[0][i + 1] + p[1][i];
  }
};

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

int stencil() {
  lib::halo_bounds hb(1);
  mhp::distributed_vector<Row> a(rows, hb), b(rows, hb);
  mhp::for_each(a, [](auto &row) { rng::iota(row, 100); });
  mhp::for_each(b, [](auto &row) { rng::fill(row, 0); });

  auto in = rng::subrange(a.begin() + 1, a.end() - 1);
  auto out = rng::subrange(b.begin() + 1, b.end() - 1);
  fmt::print("a.size() {}\n", a.size());
  fmt::print("in.size() {}\n", in.size());
  for (std::size_t s = 0; s < steps; s++) {
    mhp::halo(in).exchange();
    auto x = rng::views::zip(in, out);
    auto x_seg = lib::ranges::segments(x);
    fmt::print("x.size() {}\n"
               "x_seg.size() {}\n"
               "x_seg[0]/size() {}\n",
               std::size_t(x.size()), std::size_t(x_seg.size()),
               std::size_t(x_seg[0].size()));
    mhp::for_each(rng::views::zip(in, out), stencil_op);
    std::swap(in, out);
  }

  if (comm_rank == 0) {
    return check(steps % 2 ? b : a);
  } else {
    return 0;
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  mhp::init();

  cxxopts::Options options_spec(argv[0], "stencil 1d");
  // clang-format off
  options_spec.add_options()
    ("rows", "Number of rows", cxxopts::value<std::size_t>()->default_value("10"))
    ("steps", "Number of time steps", cxxopts::value<std::size_t>()->default_value("5"))
    ("help", "Print help");
  // clang-format on

  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  rows = options["rows"].as<std::size_t>();
  steps = options["steps"].as<std::size_t>();
  auto error = stencil();

  MPI_Finalize();
  return error;
}
