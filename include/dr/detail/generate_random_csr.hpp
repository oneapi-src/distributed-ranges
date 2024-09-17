// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <dr/views/csr_matrix_view.hpp>
#include <unordered_set>
#include <random>
#include <fmt/core.h>

namespace dr {

namespace {

template <typename T> struct uniform_distribution {
  using type = std::uniform_int_distribution<T>;
};

template <std::floating_point T> struct uniform_distribution<T> {
  using type = std::uniform_real_distribution<T>;
};

template <typename T>
using uniform_distribution_t = typename uniform_distribution<T>::type;

struct pair_hash {
    template <std::integral I>
    inline std::size_t operator()(const std::pair<I,I> & v) const {
        return v.first*31+v.second;
    }
};

} // namespace


template <typename T = float, std::integral I = std::size_t>
auto generate_random_csr(dr::index<I> shape, double density = 0.01,
                         unsigned int seed = 0) {

  assert(density >= 0.0 && density < 1.0);

  std::unordered_set<std::pair<I, I>, pair_hash> tuples{};
  std::vector<std::pair<std::pair<I,I>, T>> entries;
  std::size_t nnz = density * shape[0] * shape[1];
  entries.reserve(nnz);

  std::mt19937 gen(seed);
  std::uniform_int_distribution<I> row(0, shape[0] - 1);
  std::uniform_int_distribution<I> column(0, shape[1] - 1);

  uniform_distribution_t<T> value_gen(0, 1);

  while (tuples.size() < nnz) {
    auto i = row(gen);
    auto j = column(gen);
    if (tuples.find({i, j}) == tuples.end()) {
      T value = value_gen(gen);
      tuples.insert({i, j});
      entries.push_back({{i, j}, value});
    }
  }
  T *values = new T[nnz];
  I *rowptr = new I[shape[0] + 1];
  I *colind = new I[nnz];

  rowptr[0] = 0;

  std::size_t r = 0;
  std::size_t c = 0;
  std::sort(entries.begin(), entries.end());
  for (auto iter = entries.begin(); iter != entries.end(); ++iter) {
    auto &&[index, value] = *iter;
    auto &&[i, j] = index;

    values[c] = value;
    colind[c] = j;

    while (r < i) {
      if (r + 1 > shape[0]) {
        // TODO: exception?
        // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
      }
      rowptr[r + 1] = c;
      r++;
    }
    c++;

    if (c > nnz) {
      // TODO: exception?
      // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
    }
  }

  for (; r < shape[0]; r++) {
    rowptr[r + 1] = nnz;
  }

  return dr::views::csr_matrix_view(values, rowptr, colind, shape, nnz, 0);
}

template <typename T = float, std::integral I = std::size_t>
auto generate_band_csr(I size, std::size_t up_band = 3,
                         std::size_t down_band = 3) {
  std::size_t nnz = (1 + up_band + down_band) * size - (up_band * (up_band - 1) / 2) - (down_band * (down_band - 1) / 2);

  T *values = new T[nnz];
  I *rowptr = new I[size + 1];
  I *colind = new I[nnz];

  rowptr[0] = 0;

  std::size_t r = 0;
  std::size_t c = 0;
  for (auto i = 0; i < size; i++) {
    for (auto j = i - down_band; j < i ; j++) {
        if (j < 0) {
          continue;
        }
        values[c] = 1;
        colind[c] = j;
        c++;
    }
    values[c] = 1;
    colind[c] = i;
    c++;
    for (auto j = i + 1; j <= i + up_band ; j++) {
        if (j >= size) {
          continue;
        }
        values[c] = 1;
        colind[c] = j;
        c++;
    }
    rowptr[r + 1] = c + 1;
    r++;

  }

  for (; r < size; r++) {
    rowptr[r + 1] = nnz;
  }

  return dr::views::csr_matrix_view<T,I>(values, rowptr, colind, {size, size}, nnz, 0);
}

} // namespace dr
