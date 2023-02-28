// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using A = mhp::sycl_shared_allocator<T>;
using DV = mhp::distributed_vector<T, A>;

struct negate {
  void operator()(auto &&v) { v = -v; }
};

TEST(MhpTests, ForEach) {
  std::size_t n = 10;

  auto neg = [](auto &v) { v = -v; };
  DV dv_a(n);
  mhp::iota(dv_a, 100);
  mhp::for_each(mhp::device_policy(), dv_a, neg);

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(a, negate{});
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}
