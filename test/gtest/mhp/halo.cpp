#include "xhp-tests.hpp"
#include <dr/mhp/views/sliding.hpp>

template <typename T> class Halo : public testing::Test {};

TYPED_TEST_SUITE(Halo, AllTypes);

TYPED_TEST(Halo, dv_different_halos) {
  TypeParam dv(6, dr::mhp::distribution().halo(1, 2));
  std::vector<int> v(6);

  iota(dv, 1);
  std::iota(v.begin(), v.end(), 1);
  dv.halo().exchange();

  EXPECT_TRUE(equal(dv, v));
}

TYPED_TEST(Halo, dv_halos_next_0) {
  TypeParam dv(6, dr::mhp::distribution().halo(3, 0));
  std::vector<int> v(6);

  iota(dv, 1);
  std::iota(v.begin(), v.end(), 1);
  dv.halo().exchange();

  EXPECT_TRUE(equal(dv, v));
}

TYPED_TEST(Halo, dv_halos_prev_0) {
  TypeParam dv(6, dr::mhp::distribution().halo(0, 3));
  std::vector<int> v(6);

  iota(dv, 1);
  std::iota(v.begin(), v.end(), 1);
  dv.halo().exchange();

  EXPECT_TRUE(equal(dv, v));
}
