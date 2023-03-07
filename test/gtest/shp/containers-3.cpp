#include "shp-tests.hpp"

using DV = shp::distributed_vector<int>;

TEST(DistributedVector_3Proc, suite_runs_on_exactly_3_devices) {
  EXPECT_EQ(shp::nprocs(), 3);
  EXPECT_EQ(std::size(shp::devices()), 3);
}

TEST(DistributedVector_3Proc, segments_sizes_in_uneven_distribution) {
  DV dv(10);
  EXPECT_EQ(rng::size(dv.segments()), 3);
  EXPECT_EQ(rng::size(dv.segments()[0]), 4);
  EXPECT_EQ(rng::size(dv.segments()[1]), 4);
  EXPECT_EQ(rng::size(dv.segments()[2]), 2);
}

TEST(DistributedVector_3Proc,
     segments_sizes_in_uneven_zeroending_distribution) {
  DV dv(4);
  EXPECT_EQ(rng::size(dv.segments()), 2);
  EXPECT_EQ(rng::size(dv.segments()[0]), 2);
  EXPECT_EQ(rng::size(dv.segments()[1]), 2);
}

TEST(DistributedVector_3Proc, segments_sizes_in_empty_vec) {
  // this is not consistent, for non-zero sizes we do not return empty segments
  // but in case of empty vec we return one empty segment, IMO it should made
  // consistent in some way
  DV dv(0);
  EXPECT_EQ(rng::size(dv.segments()), 1);
  EXPECT_EQ(rng::size(dv.segments()[0]), 0);
}

TEST(DistributedVector_3Proc, segments_sizes_in_oneitem_vec) {
  DV dv(1);
  EXPECT_EQ(rng::size(dv.segments()), 1);
  EXPECT_EQ(rng::size(dv.segments()[0]), 1);
}

TEST(DistributedVector_3Proc, segments_joint_view_same_as_all_view) {
  check_segments(DV(0));
  check_segments(DV(1));
  check_segments(DV(4));
  check_segments(DV(10));
}
