// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <dr/mp/allocator.hpp>
#include <dr/mp/containers/broadcasted_slim_matrix.hpp>
#include <dr/mp/containers/broadcasted_vector.hpp>
#include <dr/mp/containers/distributed_sparse_matrix.hpp>
#include <dr/mp/global.hpp>
#include <fmt/core.h>
#include <ranges>

namespace dr::mp {

template <typename T, typename I, rng::output_range<T> C, typename Alloc,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(int root, C &res,
          distributed_sparse_matrix<T, I, Backend, MatDistr> &a,
          broadcasted_vector<T, Alloc> b) {
  if (default_comm().rank() == root) {
    assert(a.shape().first == res.size());
    assert(a.shape().second == b.size());
  }
  a.local_gemv_and_collect(root, res, b.broadcasted_data(), 1);
}

template <typename T, typename I, rng::output_range<T> C, typename Alloc,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(int root, C &res,
          distributed_sparse_matrix<T, I, Backend, MatDistr> &a,
          broadcasted_slim_matrix<T, Alloc> b) {
  if (default_comm().rank() == root) {
    assert(a.shape().first * b.width() == res.size());
  }
  a.local_gemv_and_collect(root, res, b.broadcasted_data(), b.width());
}

template <typename T, typename I, rng::output_range<T> C, typename Alloc,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(C &res, distributed_sparse_matrix<T, I, Backend, MatDistr> &a,
          broadcasted_vector<T, Alloc> b) {
  std::vector<T> workspace(res.size());
  gemv(0, workspace, a, b);
  auto tmp = new T[res.size()];
  if (default_comm().rank() == 0) {
    std::copy(workspace.begin(), workspace.end(), tmp);
  }
  default_comm().bcast(tmp, sizeof(T) * res.size(), 0);
  std::copy(tmp, tmp + res.size(), res.begin());
  delete[] tmp;
}

template <typename T, typename I, rng::output_range<T> C, typename Alloc,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(C &res, distributed_sparse_matrix<T, I, Backend, MatDistr> &a,
          broadcasted_slim_matrix<T, Alloc> b) {
  std::vector<T> workspace(res.size());
  gemv(0, workspace, a, b);
  auto tmp = new T[res.size()];
  if (default_comm().rank() == 0) {
    std::copy(workspace.begin(), workspace.end(), tmp);
  }
  default_comm().bcast(tmp, sizeof(T) * res.size(), 0);
  std::copy(tmp, tmp + res.size(), res.begin());
  delete[] tmp;
}

} // namespace dr::mp
