// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cassert>
#include <fmt/core.h>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <dr/detail/local_csr_matrix.hpp>
#include <dr/views/csr_matrix_view.hpp>

namespace dr {

namespace __detail {

// Preconditions:
// 1) `tuples` sorted by row, column
// 2) `tuples` has shape `shape`
// 3) `tuples` has `nnz` elements
template <typename Tuples, typename Allocator>
auto convert_to_csr(Tuples &&csr_matrix, dr::index<> shape, std::size_t nnz,
                    Allocator &&allocator) {
  auto &&[v, j] = *csr_matrix.begin()->begin();

  using T = std::remove_reference_t<decltype(v)>;
  using I = std::remove_reference_t<decltype(j)>;

  typename std::allocator_traits<Allocator>::template rebind_alloc<I>
      i_allocator(allocator);

  T *values = allocator.allocate(nnz);
  I *rowptr = i_allocator.allocate(shape[0] + 1);
  I *colind = i_allocator.allocate(nnz);

  rowptr[0] = 0;

  std::size_t r = 0;
  std::size_t c = 0;
  for (auto iter = csr_matrix.begin(); iter != csr_matrix.end(); ++iter) {
    for (auto iter2 = iter->begin(); iter2 != iter->end(); ++iter2) {
      auto &&[value, j] = *iter2;

      values[c] = value;
      colind[c] = j;
      c++;
    }
    assert(r + 1 <= shape[0]);
    rowptr[r + 1] = c;
    r++;

    assert(c <= nnz);
    // throw std::runtime_error("csr_matrix_impl_: given invalid matrix");
  }

  for (; r < shape[0]; r++) {
    rowptr[r + 1] = nnz;
  }

  return dr::views::csr_matrix_view(values, rowptr, colind,
                                    dr::index<I>(shape[0], shape[1]), nnz, 0);
}

/// Read in the Matrix Market file at location `file_path` and a return
/// a coo_matrix data structure with its contents.
template <typename T, typename I = std::size_t>
inline local_csr_matrix<T, I> read_coo_matrix(std::string file_path,
                                              bool one_indexed = true) {
  using size_type = std::size_t;

  auto begin = std::chrono::high_resolution_clock::now();
  std::ifstream f;

  f.open(file_path.c_str());

  if (!f.is_open()) {
    // TODO better choice of exception.
    throw std::runtime_error("mmread: cannot open " + file_path);
  }

  std::string buf;

  // Make sure the file is matrix market matrix, coordinate, and check whether
  // it is symmetric. If the matrix is symmetric, non-diagonal elements will
  // be inserted in both (i, j) and (j, i).  Error out if skew-symmetric or
  // Hermitian.
  std::getline(f, buf);
  std::istringstream ss(buf);
  std::string item;
  ss >> item;
  if (item != "%%MatrixMarket") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "matrix") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "coordinate") {
    throw std::runtime_error(file_path +
                             " could not be parsed as a Matrix Market file.");
  }
  bool pattern;
  ss >> item;
  if (item == "pattern") {
    pattern = true;
  } else {
    pattern = false;
  }
  // TODO: do something with real vs. integer vs. pattern?
  ss >> item;
  bool symmetric;
  if (item == "general") {
    symmetric = false;
  } else if (item == "symmetric") {
    symmetric = true;
  } else {
    throw std::runtime_error(file_path + " has an unsupported matrix type");
  }

  bool outOfComments = false;
  while (!outOfComments) {
    std::getline(f, buf);

    if (buf[0] != '%') {
      outOfComments = true;
    }
  }

  I m, n, nnz;
  // std::istringstream ss(buf);
  ss.clear();
  ss.str(buf);
  ss >> m >> n >> nnz;

  // NOTE for symmetric matrices: `nnz` holds the number of stored values in
  // the matrix market file, while `matrix.nnz_` will hold the total number of
  // stored values (including "mirrored" symmetric values).
  local_csr_matrix<T, I> matrix({m, n}, nnz);

  size_type c = 0;
  while (std::getline(f, buf)) {
    I i, j;
    T v;
    std::istringstream ss(buf);
    if (!pattern) {
      ss >> i >> j >> v;
    } else {
      ss >> i >> j;
      v = T(1);
    }
    if (one_indexed) {
      i--;
      j--;
    }

    if (i >= m || j >= n) {
      throw std::runtime_error(
          "read_MatrixMarket: file has nonzero out of bounds.");
    }

    matrix.push_back(i, {v, j});

    if (symmetric && i != j) {
      matrix.push_back(j, {v, i});
    }

    c++;
    if (c > nnz) {
      throw std::runtime_error("read_MatrixMarket: error reading Matrix Market "
                               "file, file has more nonzeros than reported.");
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  fmt::print("No sort read time {}\n", duration * 1000);

  begin = std::chrono::high_resolution_clock::now();
  matrix.sort();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  fmt::print("Sort time {}\n", duration * 1000);
  f.close();

  return matrix;
}

template <typename T, typename I, typename Allocator, typename... Args>
void destroy_csr_matrix_view(dr::views::csr_matrix_view<T, I, Args...> view,
                             Allocator &&alloc) {
  alloc.deallocate(view.values_data(), view.size());
  typename std::allocator_traits<Allocator>::template rebind_alloc<I> i_alloc(
      alloc);
  i_alloc.deallocate(view.colind_data(), view.size());
  i_alloc.deallocate(view.rowptr_data(), view.shape()[0] + 1);
}

} // namespace __detail

template <typename T, typename I = std::size_t>
auto read_csr(std::string file_path, bool one_indexed = true) {
  auto begin = std::chrono::high_resolution_clock::now();
  auto m = __detail::read_coo_matrix<T, I>(file_path, one_indexed);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  fmt::print("Read time {}\n", duration * 1000);

  auto shape = m.shape();
  auto nnz = m.size();

  begin = std::chrono::high_resolution_clock::now();
  auto t = __detail::convert_to_csr(m, shape, nnz, std::allocator<T>{});
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  fmt::print("Conversion time {}\n", duration * 1000);

  return t;
}
} // namespace dr
