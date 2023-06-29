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

TYPED_TEST(Halo, dv_local) {
  TypeParam dv(6, dr::mhp::distribution().halo(1, 1));
  iota(dv, 0);

  //  if (dv.begin().local() == 0) {
  //	std::cout << " FAILED for rank " << dr::mhp::default_comm().rank() <<
  //"/"
  //			  << dr::mhp::default_comm().size() << "\n";
  //  }

  // arrays below is function depending on size of communicator-1
  std::array<int, 6> first_legal_local_index___;
  std::array<int, 6> first_illlegal_local_index;
  int X = 10000; // to mark unused value

  switch (dr::mhp::default_comm().rank()) {
  case 0:
    first_legal_local_index___ = {0, 0, 0, 0, 0, 0};
    first_illlegal_local_index = {6, 3, 2, 2, 2, 1};
  case 1:
    first_legal_local_index___ = {X, 3, 2, 2, 2, 1};
    first_illlegal_local_index = {X, 6, 4, 4, 4, 2};
  case 2:
    first_legal_local_index___ = {X, X, 4, 4, 4, 2};
    first_illlegal_local_index = {X, X, 6, 6, 6, 3};
  case 3:
    first_legal_local_index___ = {X, X, X, 6, 6, 3};
    first_illlegal_local_index = {X, X, X, 6, 6, 4};
  case 4:
    first_legal_local_index___ = {X, X, X, X, 6, 4};
    first_illlegal_local_index = {X, X, X, X, 6, 5};
  case 5:
    first_legal_local_index___ = {X, X, X, X, X, 5};
    first_illlegal_local_index = {X, X, X, X, X, 6};
  }

  const auto c = dr::mhp::default_comm().size() - 1;
  for (int idx = std::max(0, first_legal_local_index___[c] - 1);
       idx < std::min(6, first_illlegal_local_index[c] + 1); ++idx) {
    dr::drlog.debug("checking legal idx:{}\n", idx);
    EXPECT_TRUE((dv.begin() + idx).local() != 0);
    EXPECT_EQ(*(dv.begin() + idx).local(), idx);
  }

  if (first_illlegal_local_index[c] < 6) {
    EXPECT_DEATH((dv.begin() + first_illlegal_local_index[c]).local(),
                 "Assertion.*");
  }
  if (first_legal_local_index___[c] > 0) {
    EXPECT_DEATH((dv.begin() + first_legal_local_index___[c] - 1).local(),
                 "Assertion.*");
  }
}
