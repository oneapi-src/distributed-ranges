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
  std::vector<std::size_t> sizes = {
      // 1,   comm_size - 1, (comm_size - 1) * (comm_size - 1), 4, 7, 10, 23, 100, 1234};
      4, 7, 23, 100};

  for (std::size_t n : sizes) {
    V l_v = generate_random<T>(n, 100);
    DV d_v(n);
    std::cout << comm_rank << ": >>> Test size " << n << std::endl;
    for (std::size_t idx = 0; idx < n; idx++) {
      d_v[idx] = l_v[idx];
    }
    barrier();

    std::cout << comm_rank <<": >>> sort " << n << std::endl;
    std::sort(l_v.begin(), l_v.end());
    dr::mhp::sort(d_v);
    std::cout << comm_rank <<": >>> sorted " << n << std::endl;

    barrier();
    std::cout << comm_rank << ": >>> comparison " << n << std::endl;
    EXPECT_TRUE(equal(l_v, d_v));
  }
}

TEST(MhpSort, Sort_reverse) {
  std::vector<std::size_t> sizes = {
      //1,   comm_size - 1, (comm_size - 1) * (comm_size - 1), 4, 7, 10, 23, 100, 1234};
      4, 7, 10, 23};

  for (std::size_t n : sizes) {
    V l_v = generate_random<T>(n, 1000);
    DV d_v(n);

    for (std::size_t idx = 0; idx < n; idx++) {
      d_v[idx] = l_v[idx];
    }
    barrier();

    dr::mhp::sort(d_v, std::greater<T>());
    std::sort(l_v.begin(), l_v.end(), std::greater<T>());
    barrier();

    EXPECT_TRUE(equal(l_v, d_v));
  }
}