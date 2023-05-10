// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

extern MPI_Comm comm;
extern std::size_t comm_rank;
extern std::size_t comm_size;

namespace xhp = dr::mhp;

template <typename V>
concept compliant_view = rng::forward_range<V> && rng::random_access_range<V> &&
                         rng::viewable_range<V> && requires(V &v) {
                           // test one at a time so error is apparent
                           dr::ranges::segments(v);
                           dr::ranges::segments(v).begin();
                           *dr::ranges::segments(v).begin();
                           dr::ranges::rank(*dr::ranges::segments(v).begin());
                           //  dr::ranges::local(rng::begin(dr::ranges::segments(v)[0]));
                           //  dr::mhp::local_segments(v);
                         };

inline void barrier() { dr::mhp::barrier(); }
inline void fence() { dr::mhp::fence(); }

#include "common-tests.hpp"

#ifdef QUICK_TEST

// minimal testing for quick builds
using AllTypes = ::testing::Types<dr::mhp::distributed_vector<int>>;

#else

using CPUTypes = ::testing::Types<dr::mhp::distributed_vector<int>,
                                  dr::mhp::distributed_vector<float>>;

#ifdef TEST_MHP_SYCL
using AllTypes = ::testing::Types<
    dr::mhp::distributed_vector<int, dr::mhp::sycl_shared_allocator<int>>,
    dr::mhp::distributed_vector<float, dr::mhp::sycl_shared_allocator<float>>>;
#else
using AllTypes = CPUTypes;
#endif

#endif

namespace dr::mhp {

template <typename DV>
inline std::ostream &operator<<(std::ostream &os,
                                const dv_segment<DV> &segment) {
  os << fmt::format("{}", segment);
  return os;
}

} // namespace dr::mhp
