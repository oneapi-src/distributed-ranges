// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

TEST(BroadcastedVector, BroadcastData) {
  std::size_t n = 100;
  auto rank = dr::mp::default_comm().rank();
  std::vector<int> data(n);
  if (rank == 0) {
    for (int i = 0; i < n; i++) {
      data[i] = i;
    }
  }
  dr::mp::broadcasted_vector<int> broadcasted;
  if (rank == 0) {
    broadcasted.broadcast_data(n, 0, data, dr::mp::default_comm());
  } else {
    broadcasted.broadcast_data(n, 0, rng::empty_view<int>(),
                               dr::mp::default_comm());
  }

  std::vector<int> ref(n);
  for (int i = 0; i < n; i++) {
    ref[i] = i;
  }

  EXPECT_EQ(rng::subrange(broadcasted.broadcasted_data(),
                          broadcasted.broadcasted_data() + n),
            ref);
  broadcasted.destroy_data();
}

TEST(BroadcastedVector, BroadcastDataReuse) {
  std::size_t n = 100;
  auto rank = dr::mp::default_comm().rank();
  std::vector<int> data(n);
  if (rank == 0) {
    for (int i = 0; i < n; i++) {
      data[i] = i;
    }
  }
  dr::mp::broadcasted_vector<int> broadcasted;
  if (rank == 0) {
    broadcasted.broadcast_data(n, 0, data, dr::mp::default_comm());
  } else {
    broadcasted.broadcast_data(n, 0, rng::empty_view<int>(),
                               dr::mp::default_comm());
  }

  std::vector<int> ref(n);
  for (int i = 0; i < n; i++) {
    ref[i] = i;
  }

  EXPECT_EQ(rng::subrange(broadcasted.broadcasted_data(),
                          broadcasted.broadcasted_data() + n),
            ref);
  broadcasted.destroy_data();
  EXPECT_EQ(broadcasted.broadcasted_data(), nullptr);
  if (rank == 0) {
    broadcasted.broadcast_data(n, 0, data, dr::mp::default_comm());
  } else {
    broadcasted.broadcast_data(n, 0, rng::empty_view<int>(),
                               dr::mp::default_comm());
  }
  EXPECT_EQ(rng::subrange(broadcasted.broadcasted_data(),
                          broadcasted.broadcasted_data() + n),
            ref);
  broadcasted.destroy_data();
}
