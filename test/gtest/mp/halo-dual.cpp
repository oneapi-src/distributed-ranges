// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

template <typename T> class HaloDual : public testing::Test {};

TYPED_TEST_SUITE(HaloDual, ::testing::Types<dr::mp::dual_distributed_vector<int>>);

template <typename DV>
void local_is_accessible_in_halo_region(const int halo_prev,
                                        const int halo_next) {

  DV dv(12, dr::mp::distribution().halo(halo_prev, halo_next));
  DRLOG("local_is_accessible_in_halo_region TEST START, prev:{}, next:{}",
        halo_prev, halo_next);
  iota(dv, 0);
  DRLOG("exchange start");

  std::vector<int*> datas = { dv.data(0), dv.data(100) };
  std::cout << "[datas before exchange]\n";
  for (const auto& data: datas) {
    for (size_t i = 0; i < dv.data_size(); i++) {
      std::cout << data[i] << " ";
    }
    std::cout << "\n";
  }
  
  std::cout << "[started exchange]\n";
  dv.halo().exchange();
  std::cout << "[finished exchange]\n";

  std::cout << "[datas after exchange]\n";
  for (const auto& data: datas) {
    for (size_t i = 0; i < dv.data_size(); i++) {
      std::cout << data[i] << " ";
    }
    std::cout << "\n";
  }

  // 0   1   1   0
  // 012 345 678 9ab

  // 0  1  2  2  1  0
  // 01 23 45 67 89 ab

  // 0  1  2  3  3  2
  // 01 23 45 67 89 ab

  // arrays below is function depending on size of communicator-1
  std::array<int, 6> first_local_index____;
  std::array<int, 6> first_nonlocal_index_;
  std::array<int, 6> second_local_index___;
  std::array<int, 6> second_nonlocal_index;
  const int X = 10000; // to mark unused value

  switch (dr::mp::default_comm().rank()) {
  case 0:
    first_local_index____ = {+0, +0, +0, 0, 0, 0};
    first_nonlocal_index_ = {12, +3, +2, 2, 1, 1};
    second_local_index___ = {+X, +9, 10, X, 0, 0};
    second_nonlocal_index = {+X, 12, 12, X, 1, 1};
    break;
  case 1:
    first_local_index____ = {X, 3, +2, 2, 2, 1};
    first_nonlocal_index_ = {X, 9, +4, 4, 4, 2};
    second_local_index___ = {X, X, +8, X, 0, 0};
    second_nonlocal_index = {X, X, 10, X, 1, 1};
    break;
  case 2:
    first_local_index____ = {X, X, 4, +4, 4, 2};
    first_nonlocal_index_ = {X, X, 8, +6, 6, 3};
    second_local_index___ = {X, X, X, 10, 0, 0};
    second_nonlocal_index = {X, X, X, 12, 1, 1};
    break;
  case 3:
    first_local_index____ = {X, X, X, +6, 6, 3};
    first_nonlocal_index_ = {X, X, X, 10, 6, 4};
    second_local_index___ = {X, X, X, +X, 0, 0};
    second_nonlocal_index = {X, X, X, +X, 1, 1};
    break;
  case 4:
    first_local_index____ = {X, X, X, X, 6, 4};
    first_nonlocal_index_ = {X, X, X, X, 6, 5};
    break;
  case 5:
    first_local_index____ = {X, X, X, X, X, 5};
    first_nonlocal_index_ = {X, X, X, X, X, 6};
    break;
  default:
    first_local_index____ = {X, X, X, X, X, X};
    first_nonlocal_index_ = {X, X, X, X, X, X};
  }

  const auto c = dr::mp::default_comm().size() - 1;
  auto first_legal_idx = std::max(0, first_local_index____[c] - halo_prev);
  auto first_illegal_idx = std::min(12, first_nonlocal_index_[c] + halo_next);
  auto second_legal_idx = std::max(0, second_local_index___[c] - halo_prev);
  auto second_illegal_idx = std::min(12, second_nonlocal_index[c] + halo_next);

  std::cout << "first_legal_idx, first_illegal_idx: " << first_legal_idx << " " << first_illegal_idx << "\n";
  std::cout << "second_legal_idx, second_illegal_idx: " << second_legal_idx << " " << second_illegal_idx << "\n";

  DRLOG("checking access to idx between first legal {} and first illegal {}, "
        "c:{}",
        first_legal_idx, first_illegal_idx, c);

  for (int idx = first_legal_idx; idx < first_illegal_idx; ++idx) {
    std::cout << "[idx = " << idx << "]\n";
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

  DRLOG("checking access to idx between second legal {} and second illegal {}, "
    "c:{}",
    second_legal_idx, second_illegal_idx, c);

  for (int idx = second_legal_idx; idx < second_illegal_idx; ++idx) {
    std::cout << "[idx = " << idx << "]\n";
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

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_11) {
  local_is_accessible_in_halo_region<TypeParam>(1, 1);
}

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_10) {
  local_is_accessible_in_halo_region<TypeParam>(1, 0);
}

TYPED_TEST(HaloDual, local_is_accessible_in_halo_region_halo_01) {
  local_is_accessible_in_halo_region<TypeParam>(0, 1);
}
