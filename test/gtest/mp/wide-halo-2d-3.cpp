// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"

template <typename T> class WideHalo3 : public testing::Test {};

using T = int;
using Array = dr::mp::distributed_mdarray<T, 2>;

const std::size_t redundancy = 2;
const std::array<std::size_t, 2> size = {6, 6};

dr::mp::distribution get_distribution() {
  return dr::mp::distribution()
      .halo(1)
      .redundancy(redundancy);
}

//int& get(Array& v, std::size_t i, std::size_t j) {
//  return *(v.begin() + i * 6 + j).local();
//}
//
//const int& get(const Array& v, std::size_t i, std::size_t j) {
//  return *(v.begin() + i * 6 + j).local();
//}

TEST(WideHalo3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mp::default_comm().size(), 3);
}

TEST(WideHalo3, halo2d_is_visible_after_exchange_not_earlier) {
  dr::mp::distribution dist = get_distribution();
  Array dv(size, dist);
  Array dv_out(size, dist);

  fill(dv, 1);
  fill(dv_out, 1);
  dv.halo().exchange();
  dv_out.halo().exchange();

  auto transform = [&]{
    stencil_for_each_extended<2>([](auto stencils){
      auto [x, x_out] = stencils;
      x_out(0, 0) = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          x_out(0, 0) += x(i, j);
        }
      }
    }, {1, 1}, {1, 1}, dv, dv_out);
    stencil_for_each_extended<2>([](auto stencils){
      auto [x, x_out] = stencils;
      x(0, 0) = x_out(0, 0);
    }, {0, 0}, {0, 0}, dv, dv_out);
  };
  auto print = [](std::string s, const auto& v) {
    std::cout << s << "\n";
    for (auto seg : v.segments()) {
      auto [beg, end] = seg.stencil({0, 0}, {0, 0});
      for (std::size_t i = beg[0]; i < end[0]; i++) {
        for (std::size_t j = beg[1]; j < end[1]; j++) {
          std::cout << seg.mdspan_extended()(i, j) << "\t";
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n";
  };

  print("dv", dv);
  transform();
  print("dv", dv);
  transform();
  print("dv", dv);
  dv.halo().exchange();
  dv_out.halo().exchange();
  print("dv", dv);
}

TEST(WideHalo3, halo2d_api_works) {
  dr::mp::distribution dist = get_distribution();
  Array dv(size, dist);
  Array dv_out(size, dist);

  fill(dv, 1);
  fill(dv_out, 1);
  dv.halo().exchange();
  dv_out.halo().exchange();

  auto print = [](std::string s, const auto& v) {
    std::cout << s << "\n";
    for (auto seg : v.segments()) {
      auto [beg, end] = seg.stencil({0, 0}, {0, 0});
      for (std::size_t i = beg[0]; i < end[0]; i++) {
        for (std::size_t j = beg[1]; j < end[1]; j++) {
          std::cout << seg.mdspan_extended()(i, j) << "\t";
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n";
  };

  print("dv", dv);

  halo_exchange([](Array& dv, Array& dv_out){
    stencil_for_each_extended<2>([](auto stencils){
      auto [x, x_out] = stencils;
      x_out(0, 0) = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          x_out(0, 0) += x(i, j);
        }
      }
    }, {1, 1}, {1, 1}, dv, dv_out);
    stencil_for_each_extended<2>([](auto stencils){
      auto [x, x_out] = stencils;
      x(0, 0) = x_out(0, 0);
    }, {0, 0}, {0, 0}, dv, dv_out);
  }, dv, dv_out);

  print("dv", dv);
}
