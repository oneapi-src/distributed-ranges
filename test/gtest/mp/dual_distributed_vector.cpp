// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

using T = int;
using DV = dr::mp::dual_distributed_vector<T>;
using DVI = typename DV::iterator;

TEST(MpTests, DualDistributedVectorQuery) {
  const int n = 10;
  DV a(n);

  EXPECT_EQ(a.size(), n);
}

TEST(MpTests, DualDistributedVectorIndex) {
  const std::size_t n = 10;
  DV dv(n);

  if (comm_rank == 0) {
    for (std::size_t i = 0; i < n; i++) {
      dv[i] = i + 10;
    }
  }
  dr::mp::fence();

  for (std::size_t i = 0; i < n; i++) {
    EXPECT_EQ(dv[i], i + 10);
  }

  DV dv2(n);

  if (comm_rank == 0) {
    dv2[3] = 1000;
    dv2[3] = dv[3];
  }
  dr::mp::fence();
  EXPECT_EQ(dv2[3], dv[3]);
}

TEST(MpTests, DualDistributedVectorAlgorithms) {
  const std::size_t n = 10;
  const int root = 0;
  DV dv(n);

  if (comm_rank == root) {
    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 1);

    std::iota(dv.begin(), dv.end(), 1);

    EXPECT_TRUE(equal_gtest(dv, ref));

    std::iota(ref.begin(), ref.end(), 11);
    std::copy(ref.begin(), ref.end(), dv.begin());
    EXPECT_TRUE(equal_gtest(dv, ref));

    std::iota(ref.begin(), ref.end(), 21);
    rng::copy(ref, dv.begin());
    EXPECT_TRUE(equal_gtest(dv, ref));

    std::iota(dv.begin(), dv.end(), 31);
    rng::copy(dv, ref.begin());
    EXPECT_TRUE(equal_gtest(dv, ref));
  }
}

int aa;

// Operations on a const distributed_vector
void common_operations(auto &dv) {
  aa = dv[1];
  EXPECT_EQ(dv[1], 101);
  EXPECT_EQ(*(&(dv[1])), 101);

  auto p = &dv[1];
  EXPECT_EQ(*(p + 1), 102);
}

TEST(MpTests, DualDistributedVectorReference) {
  std::size_t n = 10;
  DV dv(n);
  if (comm_rank == 0) {
    rng::iota(dv, 100);
  }
  dr::mp::fence();

  std::cout << "printing the vec\n\t[" << dv[0];
  for (std::size_t i = 1; i < n; i++) {
    std::cout << ", " << dv[i];
  }
  std::cout << "]\n";

  std::cout << "printing the vec iteratively\n\t[" << dv[0];
  for (auto iter = dv.begin(); iter != dv.end(); iter++) {
    std::cout << ", " << *iter;
  }
  std::cout << "]\n";

  const DV &cdv = dv;
  if (comm_rank == 0) {
    common_operations(cdv);
    common_operations(dv);
  }
  MPI_Barrier(comm);

  if (comm_rank == 0) {
    dv[2] = 2;
  }
  dr::mp::fence();
  EXPECT_EQ(dv[2], 2);
}

TEST(MpTests, DualDistributedVectorGranularity) {
  std::size_t gran = 3;
  std::size_t n = gran * 6;
  auto dist = dr::mp::distribution().granularity(gran);
  DV dv(n, dist);

  std::size_t previous_size = gran;
  for (auto &segment : dr::ranges::segments(dv)) {
    EXPECT_EQ(previous_size % gran, 0);
    previous_size = segment.size();
  }
}
