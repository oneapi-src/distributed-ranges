// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DV = dr::mhp::distributed_vector<T>;
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

TEST(MhpTests, DistributedVectorGranularity) {
  std::size_t gran = 3;
  std::size_t n = gran * 6;
  auto dist = dr::mhp::distribution().granularity(gran);
  DV dv(n, dist);

  std::size_t previous_size = gran;
  for (auto &segment : dr::ranges::segments(dv)) {
    EXPECT_EQ(previous_size % gran, 0);
    previous_size = segment.size();
  }
}

TEST(MhpTests, DistributedVectorSegments) {
  std::size_t segment_size = 2;
  std::size_t n = segment_size * 5;
  auto dist = dr::mhp::distribution().granularity(2);
  DV dv(n, dist, segment_size);

  EXPECT_EQ(dv.segments().size(), 5);
}

TEST(MhpTests, DistributedVectorSegmentSize) {
  std::size_t segment_size = 2;
  std::size_t n = segment_size * 5;
  auto dist = dr::mhp::distribution().granularity(2);
  DV dv(n, dist, segment_size);

  EXPECT_EQ(rng::size(dv.segments()[0]), segment_size);
}

TEST(MhpTests, DistributedVectorSegmentSizeIndex) {
  const std::size_t n = 10;
  std::size_t segment_size = 2;
  auto dist = dr::mhp::distribution().granularity(2);
  DV dv(n, dist, segment_size);

  if (comm_rank == 0) {
    for (std::size_t i = 0; i < n; i++) {
      dv[i] = i + 10;
    }
  }
  dr::mhp::fence();

  for (std::size_t i = 0; i < n; i++) {
    EXPECT_EQ(dv[i], i + 10);
  }
}

TEST(MhpTests, DistributedVectorSegmentSizeAlgorithms) {
  const std::size_t n = 10;
  const std::size_t seg_size = 2;
  auto dist = dr::mhp::distribution().granularity(2);
  const int root = 0;
  DV dv(n, dist, seg_size);

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

TEST(MhpTests, DistributedVectorSegmentSizeCompare) {
  const std::size_t n = 12;
  const std::size_t seg_size1 = 2;
  const std::size_t seg_size2 = 4;
  auto dist1 = dr::mhp::distribution().granularity(2);
  auto dist2 = dr::mhp::distribution().granularity(2);
  DV dv1(n, dist1, seg_size1);
  DV dv2(n, dist2, seg_size2);

  std::iota(dv1.begin(), dv1.end(), 0);
  std::iota(dv2.begin(), dv2.end(), 0);

  for (std::size_t i = 0; i < n; i++) {
    EXPECT_EQ(dv1[i], dv2[i]);
  }

  EXPECT_EQ(dv1.begin() + 2, dv2.begin() + 2);
}
