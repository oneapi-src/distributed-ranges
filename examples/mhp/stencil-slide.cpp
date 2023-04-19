// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "dr/mhp.hpp"

#include "cxxopts.hpp"

cxxopts::ParseResult options;

auto stencil_op = [](auto &&r) { return r[0] + r[1] + r[2]; };

int stencil(auto n, auto steps) {
  std::vector<int> a(n), b(n);
  rng::fill(a, 0);
  rng::fill(b, 0);

  // Input is a window
  auto in_curr = rng::views::sliding(a, 3);
  auto in_prev = rng::views::sliding(b, 3);

  // Output is an element
  auto out_curr = rng::subrange(b.begin() + 1, b.end() - 1);
  auto out_prev = rng::subrange(a.begin() + 1, a.end() - 1);

  // Initialize the input
  rng::iota(out_prev, 100);
  fmt::print("{}\n\n", a);

  for (std::size_t s = 0; s < steps; s++) {
    rng::transform(in_curr, out_curr.begin(), stencil_op);
    std::swap(in_curr, in_prev);
    std::swap(out_curr, out_prev);
    fmt::print("{}\n", s % 2 ? a : b);
  }

  return 0;
}

int main(int argc, char *argv[]) {
  cxxopts::Options options_spec(argv[0], "stencil 1d");
  // clang-format off
  options_spec.add_options()
    ("n", "Size of array", cxxopts::value<std::size_t>()->default_value("10"))
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

  return error;
}
