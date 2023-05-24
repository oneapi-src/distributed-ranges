// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "cxxopts.hpp"

#include "mpi.h"

#include "dr/mhp.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

extern void check_copy(std::size_t n, std::size_t b, std::size_t e);
extern void check_transform(std::size_t n, std::size_t b, std::size_t e);

bool is_equal(rng::range auto &&expected, rng::range auto &&actual) {
  for (auto e : rng::zip_view(expected, actual)) {
    if (e.first != e.second) {
      fmt::print("Expected: {}\n"
                 "Actual:   {}\n",
                 expected, actual);
      return false;
    }
  }

  return true;
}
