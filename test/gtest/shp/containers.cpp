// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using V = std::vector<int>;
using CV = const std::vector<int>;
using CVR = const std::vector<int> &;

using DV = shp::distributed_vector<int>;
using CDV = const shp::distributed_vector<int>;
using CDVR = const shp::distributed_vector<int> &;

TEST(ShpTests, DistributedVector) {
  static_assert(rng::random_access_range<V>);
  static_assert(rng::random_access_range<CV>);
  static_assert(rng::random_access_range<CVR>);

  static_assert(rng::random_access_range<DV>);
  static_assert(rng::random_access_range<CDV>);
  static_assert(rng::random_access_range<CDVR>);
}

TEST(ShpTests, DistributedVectorSegments) {
  const int n = 10;
  DV dv_a(n);
  std::iota(dv_a.begin(), dv_a.end(), 20);

  auto second = dv_a.begin() + 2;
  EXPECT_EQ(second[0], lib::ranges::segments(second)[0][0]);
}

TEST(ShpTests, DistributedVectorAllSegmentsSizeExceptLast) {
  const int n = 10;
  DV dv_a(n);
  auto segments = dv_a.segments();
  std::size_t expected_segment_size = (n + shp::nprocs() - 1) / shp::nprocs();
  std::size_t segments_number =
      n / expected_segment_size + (n % expected_segment_size != 0);

  for (int i = 0; i < segments_number - 1; i++) {
    EXPECT_EQ(segments[i].size(), expected_segment_size);
  }
}

TEST(ShpTests, DistributedVectorSegmentsNumber) {
  const int n = 10;
  DV dv_a(n);
  auto segments = dv_a.segments();
  auto segments_number = segments.size();
  std::size_t expected_segment_size = (n + shp::nprocs() - 1) / shp::nprocs();
  std::size_t expected_segments_number =
      n / expected_segment_size + (n % expected_segment_size != 0);
  EXPECT_EQ(segments_number, expected_segments_number);
}

TEST(ShpTests, DistributedVectorIterator) {
  const int n = 10;
  DV dv_a(n);
  V v_a(n);

  std::iota(dv_a.begin(), dv_a.end(), 20);
  std::iota(v_a.begin(), v_a.end(), 20);

  EXPECT_TRUE(std::equal(v_a.begin(), v_a.end(), dv_a.begin()));
}

TEST(ShpTests, DistributedVectorLastSegmentSize) {
  const size_t n = 10;
  DV dv_a(n);
  auto segments = dv_a.segments();
  std::size_t segment_size = (n + shp::nprocs() - 1) / shp::nprocs();
  std::size_t segments_number = n / segment_size + (n % segment_size != 0);
  std::size_t expected_last_element_size =
      (n % segment_size) ? n % segment_size : segment_size;
  EXPECT_EQ(segments[segments_number - 1].size(), expected_last_element_size);
}
