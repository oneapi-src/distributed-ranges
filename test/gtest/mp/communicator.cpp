// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename T> class Communicator : public testing::Test {};

using T = int;
using DV = dr::mp::distributed_vector<T>;

TEST(Communicator, Alltoallv) {
  if (comm_size <= 1)
    return;

  const std::size_t SIZE = 2;
  std::vector<T> vec_src(comm_size * SIZE);
  std::vector<T> vec_dst(comm_size * SIZE);

  rng::fill(vec_src, comm_rank * 10 + 1);

  std::vector<std::size_t> sendsizes(comm_size, SIZE);
  std::vector<std::size_t> recvsizes(comm_size, SIZE);

  std::vector<std::size_t> senddispl(comm_size);
  std::vector<std::size_t> recvdispl(comm_size);

  for (std::size_t i = 0; i < comm_size; i++) {
    senddispl[i] = recvdispl[i] = i * SIZE;
  }

  dr::mp::default_comm().alltoallv(vec_src, sendsizes, senddispl, vec_dst,
                                   recvsizes, recvdispl);

  std::vector<T> vec_ref(comm_size * SIZE);

  for (std::size_t i = 0; i < comm_size; i++)
    for (std::size_t j = 0; j < SIZE; j++) {
      vec_ref[i * SIZE + j] = 10 * i + 1;
    }

  EXPECT_TRUE(equal_gtest(vec_ref, vec_dst));
}

TEST(Communicator, Allgather) {
  if (comm_size <= 1)
    return;

  const std::size_t SIZE = 2;
  std::vector<T> vec_src(SIZE);
  std::vector<T> vec_dst(comm_size * SIZE);

  rng::fill(vec_src, comm_rank * 10 + 1);

  dr::mp::default_comm().all_gather(vec_src, vec_dst);

  std::vector<T> vec_ref(comm_size * SIZE);

  for (std::size_t i = 0; i < comm_size; i++)
    for (std::size_t j = 0; j < SIZE; j++) {
      vec_ref[i * SIZE + j] = 10 * i + 1;
    }

  EXPECT_TRUE(equal_gtest(vec_ref, vec_dst));
}
