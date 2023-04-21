// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DV = dr::mhp::distributed_vector<T>;
using A = std::allocator<T>;
using DVA = dr::mhp::distributed_vector<T, A>;
using DVI = typename DV::iterator;

TEST(MhpTests, DistributedVectorQuery) {
  const int n = 10;
  DV a(n);

  EXPECT_EQ(a.size(), n);
}

TEST(MhpTests, DistributedVectorIndex) {
  const std::size_t n = 10;
  DV dv(n);

  if (comm_rank == 0) {
    for (std::size_t i = 0; i < n; i++) {
      dv[i] = i + 10;
    }
  }
  dr::mhp::fence();

  for (std::size_t i = 0; i < n; i++) {
    EXPECT_EQ(dv[i], i + 10);
  }

  DV dv2(n);

  if (comm_rank == 0) {
    dv2[3] = 1000;
    dv2[3] = dv[3];
  }
  dr::mhp::fence();
  EXPECT_EQ(dv2[3], dv[3]);
}

TEST(MhpTests, DistributedVectorAlgorithms) {
  const std::size_t n = 10;
  const int root = 0;
  DV dv(n);

  if (comm_rank == root) {
    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 1);

    std::iota(dv.begin(), dv.end(), 1);

    EXPECT_TRUE(equal(dv, ref));

    std::iota(ref.begin(), ref.end(), 11);
    std::copy(ref.begin(), ref.end(), dv.begin());
    EXPECT_TRUE(equal(dv, ref));

    std::iota(ref.begin(), ref.end(), 21);
    rng::copy(ref, dv.begin());
    EXPECT_TRUE(equal(dv, ref));

    std::iota(dv.begin(), dv.end(), 31);
    rng::copy(dv, ref.begin());
    EXPECT_TRUE(equal(dv, ref));
  }
}

int a;

// Operations on a const distributed_vector
void common_operations(auto &dv) {
  a = dv[1];
  EXPECT_EQ(dv[1], 101);
  EXPECT_EQ(*(&(dv[1])), 101);

  auto p = &dv[1];
  EXPECT_EQ(*(p + 1), 102);
}

TEST(MhpTests, DistributedVectorReference) {
  std::size_t n = 10;
  DV dv(n);
  if (comm_rank == 0) {
    rng::iota(dv, 100);
  }
  dr::mhp::fence();

  const DV &cdv = dv;
  if (comm_rank == 0) {
    common_operations(cdv);
    common_operations(dv);
  }
  MPI_Barrier(comm);

  if (comm_rank == 0) {
    dv[2] = 2;
  }
  dr::mhp::fence();
  EXPECT_EQ(dv[2], 2);
}

#if 0
TEST(MhpTests, DistributedVectorAllocator) {
  std::size_t n = 10;
  DVA dv(n, dr::halo_bounds(0), std::allocator<T>{});
  dr::mhp::fill(dv, 22);
  if (comm_rank == 0) {
    std::vector<T> v(n);
    rng::fill(v, 22);
    EXPECT_TRUE(equal(dv, v));
  }
}
#endif
