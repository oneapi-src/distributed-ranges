// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu-tests.hpp"

using V = std::vector<int>;
using DV = lib::distributed_vector<int>;

void check_copy(std::size_t n, std::size_t b, std::size_t e) {
  V v_in(n), v(n), v1(n), v2(n);
  rng::iota(v_in, 100);

  lib::distributed_vector<int> dv_in(n), dv1(n), dv2(n), dv3(n), dv4(n), dv5(n),
      dv6(n), dv7(n), dv8(n), dv9(n);
  lib::iota(dv_in, 100);
  lib::copy(dv_in.begin() + b, dv_in.begin() + e, dv1.begin() + b);
  lib::copy(rng::subrange(dv_in.begin() + b, dv_in.begin() + e),
            dv2.begin() + b);

  lib::copy(0, v_in.begin() + b, v_in.begin() + e, dv4.begin() + b);
  lib::copy(0, rng::subrange(v_in.begin() + b, v_in.begin() + e),
            dv6.begin() + b);
  if (comm_rank == 0) {
    lib::copy(0, &*(v_in.begin() + b), e - b, dv7.begin() + b);
  } else {
    lib::copy(0, nullptr, e - b, dv7.begin() + b);
  }
  lib::copy(0, comm_rank == 0 ? &*(v_in.begin() + b) : nullptr, e - b,
            dv5.begin() + b);

  lib::copy(0, dv_in.begin() + b, dv_in.begin() + e, v1.begin() + b);
  lib::copy(0, dv_in.begin() + b, e - b,
            comm_rank == 0 ? &*(v2.begin() + b) : nullptr);

  if (comm_rank == 0) {
    std::copy(dv_in.begin() + b, dv_in.begin() + e, dv3.begin() + b);
  }
  if (comm_rank == 0) {
    lib::copy(0, v_in.begin() + b, v_in.begin() + e, dv8.begin() + b);
  } else {
    lib::copy(0, nullptr, nullptr, dv8.begin() + b);
  }

  lib::copy(0, comm_rank == 0 ? &*(v_in.begin() + b) : nullptr,
            comm_rank == 0 ? &*(v_in.begin() + e) : nullptr, dv9.begin() + b);

  dv3.fence();

  if (comm_rank == 0) {
    std::copy(v_in.begin() + b, v_in.begin() + e, v.begin() + b);

    EXPECT_TRUE(equal(dv1, v));
    EXPECT_TRUE(equal(dv2, v));
    EXPECT_TRUE(equal(dv3, v));
    EXPECT_TRUE(equal(dv4, v));
    EXPECT_TRUE(equal(dv5, v));
    EXPECT_TRUE(equal(dv6, v));
    EXPECT_TRUE(equal(dv7, v));
    EXPECT_TRUE(equal(dv8, v));
    EXPECT_TRUE(equal(dv9, v));

    EXPECT_TRUE(equal(v1, v));
    EXPECT_TRUE(equal(v2, v));
  }
}

TEST(CpuMpiTests, CopyDistributedVector) {
  std::size_t n = 10;

  check_copy(n, 0, n);
  check_copy(n, n / 2 - 1, n / 2 + 1);
}

struct negate {
  void operator()(auto &v) { v = -v; }
};

TEST(CpuMpiTests, ForEach) {
  std::size_t n = 10;

  V a(n), a_in(n);
  rng::iota(a, 100);
  rng::iota(a_in, 100);
  rng::for_each(a, negate{});

  DV dv_a(n);
  lib::iota(dv_a, 100);
  lib::for_each(dv_a, negate{});
  EXPECT_TRUE(unary_check(a_in, a, dv_a));
}
