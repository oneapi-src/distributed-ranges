// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <dr/mp/allocator.hpp>
#include <dr/mp/containers/distributed_sparse_matrix.hpp>
#include <dr/mp/global.hpp>
#include <fmt/core.h>
#include <ranges>

namespace dr::mp {

template <typename T, typename I, rng::output_range<T> C, rng::input_range B,
          typename Backend, typename MatDistr>
  requires(vector_multiplicable<MatDistr>)
void gemv(int root, C &res,
          distributed_sparse_matrix<T, I, Backend, MatDistr> &a, B &b) {
  if (default_comm().rank() == root) {
    assert(a.shape().first == res.size());
    assert(a.shape().second == b.size());
  }
  // copy b to all machines
  auto communicator = default_comm();
  __detail::allocator<T> alloc;
  auto broadcasted_b = alloc.allocate(a.shape().second);
  if (communicator.rank() == root) {
    rng::copy(b.begin(), b.end(), broadcasted_b);
  }
  communicator.bcast(broadcasted_b, a.shape().second * sizeof(T), root);
  a.local_gemv_and_collect(root, res, broadcasted_b);
  alloc.deallocate(broadcasted_b, a.shape().second);
  // a.fence();
  // if (default_comm().rank() == root) {
  //     for (int i = 0; i < a.shape().first; i++) {
  //         fmt::print("Result {} {}\n", i, res[i]);
  //     }
  // }
}

} // namespace dr::mp
