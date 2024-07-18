// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename T> class Communicator3 : public testing::Test {};

using T = int;

TYPED_TEST_SUITE(Communicator3, AllTypes);

TYPED_TEST(Communicator3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mp::default_comm().size(), 3);
}

TEST(Communicator3, AlltoallvThreeRanksOnly) {
  const std::size_t max_send_recv_size = 3;
  std::vector<T> vec_src = {1, 2, 3, 0, 5, 6};
  std::vector<T> vec_dst(comm_size * max_send_recv_size, 0);

  std::vector<std::size_t> sendsizes = {3, 2, 1};
  std::vector<std::size_t> recvsizes = {3, 3, 3};

  std::vector<std::size_t> senddispl = {0, 3, 5};
  std::vector<std::size_t> recvdispl = {0, 3, 6};

  dr::mp::default_comm().alltoallv(vec_src, sendsizes, senddispl, vec_dst,
                                   recvsizes, recvdispl);

  std::vector<T> vec_ref_0 = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<T> vec_ref_1 = {0, 5, 0, 0, 5, 0, 0, 5, 0};
  std::vector<T> vec_ref_2 = {6, 0, 0, 6, 0, 0, 6, 0, 0};

  switch (comm_rank) {
  case 0:
    EXPECT_EQ(vec_ref_0, vec_dst);
    break;
  case 1:
    EXPECT_EQ(vec_ref_1, vec_dst);
    break;
  case 2:
    EXPECT_EQ(vec_ref_2, vec_dst);
    break;
  }
}
