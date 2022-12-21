// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include "range/v3/all.hpp"

#include "mkl.h"

#include "utils.hpp"

void transpose(std::size_t rows, std::size_t cols, double *src, double *dst) {
  mkl_domatcopy('R', 'T', rows, cols, 1.0, src, cols, dst, rows);
}

template <typename T> struct transpose_serial {
  std::vector<T> a, b;
  std::size_t rows, cols;

  void init(std::size_t r, std::size_t c) {
    rows = r;
    cols = c;
    a.resize(rows * cols);
    b.resize(rows * cols);

    set_step(a, 0);
    set_step(b, 100);
  }

  void compute() { transpose(rows, cols, a.data(), b.data()); }

  void check(std::vector<T> &result) { assert(::check(result, b) == 0); }
};
