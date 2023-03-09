// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

extern MPI_Comm comm;
extern std::size_t comm_rank;
extern std::size_t comm_size;

namespace zhp = rng::views;
namespace xhp = mhp;

inline void barrier() { mhp::barrier(); }
inline void fence() { mhp::fence(); }

template <typename Alloc> inline auto default_policy(Alloc alloc) {
  return std::execution::par_unseq;
}

template <typename T>
inline auto default_policy(
    const mhp::distributed_vector<T, mhp::sycl_shared_allocator<T>> &dv) {
  return mhp::device_policy();
}

template <typename T>
inline auto
default_policy(const mhp::distributed_vector<T, std::allocator<T>> &dv) {
  return std::execution::par_unseq;
}

#include "common-tests.hpp"

#include "mhp/reduce.hpp"
