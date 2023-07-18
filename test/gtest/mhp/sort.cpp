// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/detail/communicator.hpp>
#include <dr/mhp/algorithms/sort.hpp>

using T = int;
using DV = dr::mhp::distributed_vector<T, dr::mhp::default_allocator<T>>;
using V = std::vector<T>;

TEST(MhpSort, Sort) {
  std::vector<std::size_t> sizes = {36};

  unsigned short int seedv[] = {1, 2, 3};
  seed48(seedv);
  for (std::size_t n : sizes) {
    std::vector<T> l_v = generate_random<T>(n, 100);

    dr::mhp::distributed_vector<T> d_v(n);

    for (int idx = 0; idx < n; idx++) {
      d_v[idx] = l_v[idx];
    }

    std::sort(l_v.begin(), l_v.end());

    dr::mhp::sort(d_v);

    // for (std::size_t i = 0; i < l_v.size(); i++) {
    //   EXPECT_EQ(l_v[i], d_v[i]);
    // }
  }
}
