// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

void check_fill(std::size_t n, std::size_t b, std::size_t size) {
  auto e = b + size;
  int val = 33;

  DV dv1(n), dv2(n);
  DV dv3(n);
  mhp::iota(dv1, 10);
  mhp::iota(dv2, 10);
  mhp::iota(dv3, 10);
  mhp::fill(dv1.begin() + b, dv1.begin() + e, val);
  mhp::fill(rng::subrange(dv3.begin() + b, dv3.begin() + e), val);

  if (comm_rank == 0) {
    std::fill(dv2.begin() + b, dv2.begin() + e, val);
  }
  dv2.fence();

  if (comm_rank == 0) {
    V v(n);
    rng::iota(v, 10);
    std::fill(v.begin() + b, v.begin() + e, val);

    EXPECT_TRUE(equal(dv1, v));
    EXPECT_TRUE(equal(dv2, v));
    EXPECT_TRUE(equal(dv3, v));
  }
}

TEST(MhpTests, Fill) {
  std::size_t n = 10;

  check_fill(n, 0, n);
  check_fill(n, n / 2 - 1, 2);
}

struct negate {
  void operator()(auto &&v) { v = -v; }
};

TEST(MhpTests, ForEach) {
  std::size_t n = 10;

  DV dv_a(n);
  mhp::iota(dv_a, 100);
  mhp::for_each(dv_a, negate{});

  if (comm_rank == 0) {
    V a(n), a_in(n);
    rng::iota(a, 100);
    rng::iota(a_in, 100);
    rng::for_each(a, negate{});
    EXPECT_TRUE(unary_check(a_in, a, dv_a));
  }
}
