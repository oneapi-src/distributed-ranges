#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Halo : public testing::Test {};

TYPED_TEST_SUITE(Halo, AllTypes);

TYPED_TEST(Halo, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mhp::default_comm().size(), 3); // dr-style ignore
}

TYPED_TEST(Halo, dv_different_halos) {
  TypeParam dv(6, dr::mhp::distribution().halo(1, 2));

  iota(dv, 1);
  dv.halo().exchange();

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(*(dv.begin() + 2), 3); // from node 1
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(*(dv.begin() + 0), 1); // from node 0
    EXPECT_EQ(*(dv.begin() + 1), 2); // from node 0

    EXPECT_EQ(*(dv.begin() + 4), 5); // from node 2
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(*(dv.begin() + 3), 4); // from node 1
    EXPECT_EQ(*(dv.begin() + 4), 5); // from node 1
  }
}

TYPED_TEST(Halo, dv_halos_next_0) {
  TypeParam dv(6, dr::mhp::distribution().halo(2, 0));
  std::vector<int> v(6);

  iota(dv, 1);
  dv.halo().exchange();

  if (dr::mhp::default_comm().rank() == 0) {
    ;
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(*(dv.begin() + 0), 1); // from node 0
    EXPECT_EQ(*(dv.begin() + 1), 2); // from node 0
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
    EXPECT_EQ(*(dv.begin() + 3), 4); // from node 1
    EXPECT_EQ(*(dv.begin() + 4), 5); // from node 1
  }
}

TYPED_TEST(Halo, dv_halos_prev_0) {
  TypeParam dv(6, dr::mhp::distribution().halo(0, 2));
  iota(dv, 1);
  dv.halo().exchange();

  if (dr::mhp::default_comm().rank() == 0) {
    EXPECT_EQ(*(dv.begin() + 2), 3); // from node 1
    EXPECT_EQ(*(dv.begin() + 3), 4); // from node 1
  } else if (dr::mhp::default_comm().rank() == 1) {
    EXPECT_EQ(*(dv.begin() + 4), 5); // from node 2
    EXPECT_EQ(*(dv.begin() + 5), 6); // from node 3
  } else {
    assert(dr::mhp::default_comm().rank() == 2);
  }
}
