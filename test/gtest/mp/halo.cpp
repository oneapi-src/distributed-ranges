// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

template <typename T> class Halo : public testing::Test {};

TYPED_TEST_SUITE(Halo, AllTypes);

template <typename DV>
void local_is_accessible_in_halo_region(const int halo_prev,
                                        const int halo_next) {

  DV dv(6, dr::mp::distribution().halo(halo_prev, halo_next));
  DRLOG("local_is_accessible_in_halo_region TEST START, prev:{}, next:{}",
        halo_prev, halo_next);
  iota(dv, 0);
  DRLOG("exchange start");
  
  // dv.halo().exchange();

  // arrays below is function depending on size of communicator-1
  std::array<int, 6> first_local_index___;
  std::array<int, 6> first_nonlocal_index;
  const int X = 10000; // to mark unused value

  switch (dr::mp::default_comm().rank()) {
  case 0:
    first_local_index___ = {0, 0, 0, 0, 0, 0};
    first_nonlocal_index = {6, 3, 2, 2, 2, 1};
    break;
  case 1:
    first_local_index___ = {X, 3, 2, 2, 2, 1};
    first_nonlocal_index = {X, 6, 4, 4, 4, 2};
    break;
  case 2:
    first_local_index___ = {X, X, 4, 4, 4, 2};
    first_nonlocal_index = {X, X, 6, 6, 6, 3};
    break;
  case 3:
    first_local_index___ = {X, X, X, 6, 6, 3};
    first_nonlocal_index = {X, X, X, 6, 6, 4};
    break;
  case 4:
    first_local_index___ = {X, X, X, X, 6, 4};
    first_nonlocal_index = {X, X, X, X, 6, 5};
    break;
  case 5:
    first_local_index___ = {X, X, X, X, X, 5};
    first_nonlocal_index = {X, X, X, X, X, 6};
    break;
  default:
    first_local_index___ = {X, X, X, X, X, X};
    first_nonlocal_index = {X, X, X, X, X, X};
  }

  const auto c = dr::mp::default_comm().size() - 1;
  auto first_legal_idx = std::max(0, first_local_index___[c] - halo_prev);
  auto first_illegal_idx = std::min(6, first_nonlocal_index[c] + halo_next);

  std::cout << "first_legal_idx: " << first_legal_idx << "\n";
  std::cout << "first_illegal_idx: " << first_illegal_idx << "\n";

  DRLOG("checking access to idx between first legal {} and first illegal {}, "
        "c:{}",
        first_legal_idx, first_illegal_idx, c);

  for (int idx = first_legal_idx; idx < first_illegal_idx; ++idx) {
    typename DV::value_type *local_ptr = (dv.begin() + idx).local();
    EXPECT_TRUE(local_ptr != nullptr);
    typename DV::value_type value_on_host;

    if (dr::mp::use_sycl())
      dr::mp::__detail::sycl_copy(local_ptr, &value_on_host);
    else
      value_on_host = *local_ptr;

    DRLOG("checking idx:{}", idx);
    EXPECT_EQ(value_on_host, idx);
  }
  DRLOG("checks ok");

  // although assertions indeed happen, but they are not caught by EXPECT_DEATH
  //  if (first_illegal_idx < 6) {
  //    dr::drlog.debug("checking first illegal idx:{} after legal ones\n",
  //                    first_illegal_idx);
  //    EXPECT_DEATH((dv.begin() + first_illegal_idx).local(), "Assertion.*");
  //  }
  //  if (first_legal_idx > 0) {
  //    dr::drlog.debug("checking last illegal idx:{} before legal ones\n",
  //                    first_legal_idx - 1);
  //    EXPECT_DEATH((dv.begin() + first_legal_idx - 1).local(), "Assertion.*");
  //  }
}

TYPED_TEST(Halo, local_is_accessible_in_halo_region_halo_11) {
  local_is_accessible_in_halo_region<TypeParam>(1, 1);
}

TYPED_TEST(Halo, local_is_accessible_in_halo_region_halo_10) {
  local_is_accessible_in_halo_region<TypeParam>(1, 0);
}

TYPED_TEST(Halo, local_is_accessible_in_halo_region_halo_01) {
  local_is_accessible_in_halo_region<TypeParam>(0, 1);
}
