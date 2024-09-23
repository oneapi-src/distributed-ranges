// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

template <typename T> class WideHalo3 : public testing::Test {};

using T = int;
using Array = dr::mp::distributed_vector<T>;

const std::size_t redundancy = 2;
const std::size_t size = 6;

dr::mp::distribution get_distribution() {
  return dr::mp::distribution()
      .halo(1)
      .redundancy(redundancy);
}

int& get(Array& v, std::size_t i) {
  return *(v.begin() + i).local();
}

TEST(WideHalo3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mp::default_comm().size(), 3);
}

TEST(WideHalo3, halo_is_visible_after_exchange_not_earlier) {
  dr::mp::distribution dist = get_distribution();
  Array dv(size, dist);
  Array dv_out(size, dist);

  fill(dv, 1);
  fill(dv_out, 1);
  dv.halo().exchange();
  dv_out.halo().exchange();

  auto print = [&](const auto& v) {
    for (auto seg : v.segments()) {
      for (auto i = seg.begin_stencil({0ul})[0]; i < seg.end_stencil({0ul})[0]; i++) {
        std::cout << *(seg.begin() + i).local() << " ";
      }
    }
    std::cout << "\n";
  };

  auto transform = [&]{
    stencil_for_each_extended<1>([](auto stencils, auto id){
      auto [x, x_out] = stencils;
      x_out(0) = x(-1) + x(0) + x(1);
    }, {1}, {1}, dv, dv_out);
    stencil_for_each_extended<1>([](auto stencils, auto id){
      auto [x, x_out] = stencils;
      x(0) = x_out(0);
    }, {0}, {0}, dv, dv_out);
  };

  transform();
  print(dv);

  // after first step, only actually stored values and their neighbours are guaranteed to be correct
  switch (dr::mp::default_comm().rank()) {
    case 0:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 3);
      EXPECT_EQ(get(dv, 2), 3);
      EXPECT_EQ(get(dv, 3), 1);
      break;
    case 1:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 3);
      EXPECT_EQ(get(dv, 2), 3);
      EXPECT_EQ(get(dv, 3), 3);
      EXPECT_EQ(get(dv, 4), 3);
      EXPECT_EQ(get(dv, 5), 1);
      break;
    case 2:
      EXPECT_EQ(get(dv, 2), 1);
      EXPECT_EQ(get(dv, 3), 3);
      EXPECT_EQ(get(dv, 4), 3);
      EXPECT_EQ(get(dv, 5), 1);
      break;
  }

  // after second step, only actually stored values are guaranteed to be correct

  transform();
  print(dv);

  switch (dr::mp::default_comm().rank()) {
    case 0:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 7);
      EXPECT_EQ(get(dv, 2), 7);
      EXPECT_EQ(get(dv, 3), 1);
      break;
    case 1:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 7);
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      EXPECT_EQ(get(dv, 4), 7);
      EXPECT_EQ(get(dv, 5), 1);
      break;
    case 2:
      EXPECT_EQ(get(dv, 2), 1);
      EXPECT_EQ(get(dv, 3), 7);
      EXPECT_EQ(get(dv, 4), 7);
      EXPECT_EQ(get(dv, 5), 1);
      break;
  }

  // after exchange all are correct
  dv.halo().exchange();
  print(dv);

  switch (dr::mp::default_comm().rank()) {
    case 0:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 7);
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      break;
    case 1:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 7);
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      EXPECT_EQ(get(dv, 4), 7);
      EXPECT_EQ(get(dv, 5), 1);
      break;
    case 2:
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      EXPECT_EQ(get(dv, 4), 7);
      EXPECT_EQ(get(dv, 5), 1);
      break;
  }
}

TEST(WideHalo3, halo_api_works) {
  dr::mp::distribution dist = get_distribution();
  Array dv(size, dist);
  Array dv_out(size, dist);

  fill(dv, 1);
  fill(dv_out, 1);
  dv.halo().exchange();
  dv_out.halo().exchange();

  halo_exchange([](Array& dv, Array& dv_out){
    stencil_for_each_extended<1>([](auto stencils, auto id){
      auto [x, x_out] = stencils;
      x_out(0) = x(-1) + x(0) + x(1);
    }, {1}, {1}, dv, dv_out);
    stencil_for_each_extended<1>([](auto stencils, auto id){
      auto [x, x_out] = stencils;
      x(0) = x_out(0);
    }, {0}, {0}, dv, dv_out);
  }, dv, dv_out);
  // after exchange all are correct
  switch (dr::mp::default_comm().rank()) {
    case 0:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 7);
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      break;
    case 1:
      EXPECT_EQ(get(dv, 0), 1);
      EXPECT_EQ(get(dv, 1), 7);
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      EXPECT_EQ(get(dv, 4), 7);
      EXPECT_EQ(get(dv, 5), 1);
      break;
    case 2:
      EXPECT_EQ(get(dv, 2), 9);
      EXPECT_EQ(get(dv, 3), 9);
      EXPECT_EQ(get(dv, 4), 7);
      EXPECT_EQ(get(dv, 5), 1);
      break;
  }
}

