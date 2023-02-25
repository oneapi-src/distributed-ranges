// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-sycl-tests.hpp"

using T = int;
using A = mhp::sycl_shared_allocator<T>;
using DV = mhp::distributed_vector<T, A>;

TEST(MhpSyclTests, DistributedVectorConstructors) {
  DV a1(10);
  mhp::iota(a1, 100);
}
