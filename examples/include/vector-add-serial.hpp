// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include "utils.hpp"

template <typename T> struct vector_add_serial {
  std::vector<T> a, b, c;
  size_t size;

  void init(size_t n) {
    size = n;
    a.resize(n);
    b.resize(n);
    c.resize(n);

    set_step(a, 0);
    set_step(b, 10, 10);
    set_step(c, 0);
  }

  void compute() {
    for (size_t i = 0; i < size; i++) {
      c[i] = a[i] + b[i];
    }
  }

  void check(std::vector<T> &result) { assert(::check(result, c) == 0); }
};
