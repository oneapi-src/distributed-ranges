// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "cxxopts.hpp"
#include "dr/mp.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>

#define TEST_MP

extern MPI_Comm comm;
extern std::size_t comm_rank;
extern std::size_t comm_size;
extern cxxopts::ParseResult options;

namespace xp = dr::mp;

template <typename V>
concept compliant_view = rng::forward_range<V> && rng::random_access_range<V> &&
                         rng::viewable_range<V> && requires(V &v) {
                           // test one at a time so error is apparent
                           dr::ranges::segments(v);
                           dr::ranges::segments(v).begin();
                           *dr::ranges::segments(v).begin();
                           dr::ranges::rank(*dr::ranges::segments(v).begin());
                           //  dr::ranges::local(rng::begin(dr::ranges::segments(v)[0]));
                           //  dr::mp::local_segments(v);
                         };

inline void barrier() { dr::mp::barrier(); }
inline void fence() { dr::mp::fence(); }
inline void fence_on(auto &&obj) { obj.fence(); }

#include "common-tests.hpp"

// minimal testing for quick builds
#ifdef DRISHMEM
using AllTypes =
    ::testing::Types<dr::mp::distributed_vector<int, dr::mp::IshmemBackend>,
                     dr::mp::distributed_vector<int>>;
using IshmemTypes =
    ::testing::Types<dr::mp::distributed_vector<int, dr::mp::IshmemBackend>>;
#else
using AllTypes = ::testing::Types<dr::mp::distributed_vector<int>,
                                  dr::mp::dual_distributed_vector<int>>;
using IshmemTypes = ::testing::Types<dr::mp::distributed_vector<int>,
                                     dr::mp::dual_distributed_vector<int>>;

#endif
using AllTypesWithoutIshmem = 
  ::testing::Types<dr::mp::distributed_vector<int>>, 
                   dr::mp::dual_distributed_vector<int>>;

namespace dr::mp {

template <typename DV>
inline std::ostream &operator<<(std::ostream &os,
                                const dv_segment<DV> &segment) {
  os << fmt::format("{}", segment);
  return os;
}

} // namespace dr::mp
