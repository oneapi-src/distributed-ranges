// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "sycl-mpi-tests.hpp"

TEST(SyclMpiTests, DplReduce) {
  sycl::queue q;
  auto device_policy = oneapi::dpl::execution::make_device_policy(q);

  using allocator = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
  allocator alloc(q);

  std::size_t n = 10;
  std::vector<int, allocator> v(n, alloc);

  int dpl_sum = std::reduce(device_policy, v.cbegin(), v.cend(), 1000);
  int sum = std::reduce(v.cbegin(), v.cend(), 1000);
  EXPECT_EQ(dpl_sum, sum);
}
