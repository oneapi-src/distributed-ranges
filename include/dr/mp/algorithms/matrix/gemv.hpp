// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <ranges>
#include <dr/mp/containers/distributed_sparse_matrix.hpp>
#include <dr/mp/global.hpp>
#include <dr/mp/allocator.hpp>
#include <fmt/core.h>

namespace dr::mp {

template <typename T, typename I, rng::output_range<T> C, rng::input_range B, typename Backend> //TODO?:, typename MatDistr>
void gemv(int root, C &res, distributed_sparse_matrix<T, I, Backend> &a, B &b) {
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

    // multiply b by local segment
    auto res_alloc = alloc.allocate(a.shape().first);
    a.local_gemv(res_alloc, broadcasted_b);

    // reduce result by adding partial results
    if (default_comm().rank() == root) {
        auto gathered_res = alloc.allocate(a.shape().first * communicator.size());
        communicator.gather(res_alloc, gathered_res, a.shape().first, root);
        rng::fill(res, 0);
        for (int i = 0; i < communicator.size(); i++) {
            auto row_bounds = a.local_row_bounds(i);
            for (int j = row_bounds.first; j < row_bounds.second; j++) {
                res[j] += gathered_res[a.shape().first * i + j - row_bounds.first];
            }
        }
        alloc.deallocate(gathered_res, a.shape().first * communicator.size());
    }
    else {
        communicator.gather(res_alloc, static_cast<T*>(nullptr), a.shape().first, root);
    }
    alloc.deallocate(broadcasted_b, a.shape().second);
    alloc.deallocate(res_alloc, a.shape().first);
    // a.fence();
    // if (default_comm().rank() == root) {
    //     for (int i = 0; i < a.shape().first; i++) {
    //         fmt::print("Result {} {}\n", i, res[i]);
    //     }
    // }

}

}