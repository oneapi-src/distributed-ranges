// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename T> class Communicator : public testing::Test {};

using T = int;
using DV = dr::mhp::distributed_vector<T>;

TEST(Communicator, Alltoallv) {
  const std::size_t SIZE = 2;
  std::vector<T> vec_src(comm_size * SIZE);
  std::vector<T> vec_dst(comm_size * SIZE);

  rng::fill(vec_src, comm_rank * 10 + 1);

  std::vector<int> sendsizes(comm_size, SIZE);
  std::vector<int> recvsizes(comm_size, SIZE);

  std::vector<int> senddispl(comm_size);
  std::vector<int> recvdispl(comm_size);

  for (std::size_t i = 0; i < comm_size; i++) {
    senddispl[i] = recvdispl[i] = i * SIZE;
  }

  dr::mhp::default_comm().alltoallv(vec_src.data(), sendsizes, senddispl,
                                    vec_dst.data(), recvsizes, recvdispl);

  std::vector<T> vec_ref(comm_size * SIZE);

  for (std::size_t i = 0; i < comm_size; i++)
    for (std::size_t j = 0; j < SIZE; j++) {
      vec_ref[i * SIZE + j] = 10 * i + 1;
    }

  EXPECT_TRUE(equal(vec_ref, vec_dst));
}

TEST(Communicator, Allgather) {
  const std::size_t SIZE = 2;
  std::vector<T> vec_src(SIZE);
  std::vector<T> vec_dst(comm_size * SIZE);

  rng::fill(vec_src, comm_rank * 10 + 1);

  dr::mhp::default_comm().all_gather(vec_src, vec_dst);

  std::vector<T> vec_ref(comm_size * SIZE);

  for (std::size_t i = 0; i < comm_size; i++)
    for (std::size_t j = 0; j < SIZE; j++) {
      vec_ref[i * SIZE + j] = 10 * i + 1;
    }

  EXPECT_TRUE(equal(vec_ref, vec_dst));
}