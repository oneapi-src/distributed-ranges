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

namespace xhp = mhp;

template <typename V>
concept compliant_view = rng::forward_range<V> && requires(V &v) {
  // test one at a time so error is apparent
  dr::ranges::segments(v);
  dr::ranges::rank(dr::ranges::segments(v)[0]);
  rng::begin(dr::ranges::segments(v)[0]);
  dr::ranges::local(rng::begin(dr::ranges::segments(v)[0]));
  mhp::local_segments(v);
};

inline void barrier() { mhp::barrier(); }
inline void fence() { mhp::fence(); }

#ifdef SYCL_LANGUAGE_VERSION
template <typename T>
inline auto default_policy(
    const mhp::distributed_vector<T, mhp::sycl_shared_allocator<T>> &dv) {
  return mhp::device_policy();
}
#endif

template <typename T>
inline auto
default_policy(const mhp::distributed_vector<T, std::allocator<T>> &dv) {
  return std::execution::par_unseq;
}

#include "common-tests.hpp"
