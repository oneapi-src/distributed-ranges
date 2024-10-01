// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <dr/mp/allocator.hpp>
#include <dr/mp/containers/distributed_sparse_matrix.hpp>
#include <dr/mp/global.hpp>
#include <fmt/core.h>
#include <ranges>
#include <dr/mp/containers/broadcasted_vector.hpp>
#include <dr/mp/containers/broadcasted_slim_matrix.hpp>


namespace dr::mp {

template <typename T, typename I, rng::output_range<T> C, typename Alloc,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(int root, C &res,
          distributed_sparse_matrix<T, I, Backend, MatDistr> &a, broadcasted_vector<T,Alloc> b) {
  if (default_comm().rank() == root) {
    assert(a.shape().first == res.size());
  }
  // copy b to all machines
  // auto communicator = default_comm();
  // __detail::allocator<T> alloc;
  // auto broadcasted_b = alloc.allocate(a.shape().second);
  // if (communicator.rank() == root) {
  //   rng::copy(b.begin(), b.end(), broadcasted_b);
  // }
  
  // communicator.bcast(broadcasted_b, a.shape().second * sizeof(T), root);
  a.local_gemv_and_collect(root, res, b.broadcasted_data(), 1);
  
  // alloc.deallocate(broadcasted_b, a.shape().second);
  // a.fence();
  // if (default_comm().rank() == root) {
  //     for (int i = 0; i < a.shape().first; i++) {
  //         fmt::print("Result {} {}\n", i, res[i]);
  //     }
  // }
}

template <typename T, typename I, rng::output_range<T> C, typename Alloc,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(int root, C &res,
          distributed_sparse_matrix<T, I, Backend, MatDistr> &a, broadcasted_slim_matrix<T,Alloc> b) {
  if (default_comm().rank() == root) {
    assert(a.shape().first * b.width() == res.size());
  }
  a.local_gemv_and_collect(root, res, b.broadcasted_data(), b.width());
}

} // namespace dr::mp
