// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using T = int;
using DV = shp::distributed_vector<T, shp::shared_allocator<T>>;
using V = std::vector<T>;

TEST(ShpTests, Iota) {
  const int n = 10;
  V a(n);
  DV dv_a(n);

  std::iota(a.begin(), a.end(), 20);
  std::iota(dv_a.begin(), dv_a.end(), 20);
  EXPECT_TRUE(equal(a, dv_a));
}

// hard to reproduce fails
TEST(ShpTests, DISABLED_InclusiveScan) {
  std::size_t n = 100;

  shp::distributed_vector<int, shp::device_allocator<int>> v(n);
  shp::distributed_vector<int, shp::device_allocator<int>> o(v.size() * 2);
  std::vector<int> lv(n);

  // Range case, no binary op or init, perfectly aligned
  for (auto &&x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v, v);

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], v[i]);
  }

  // Range case, binary op no init, non-aligned ranges
  for (auto &&x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v, o, std::plus<>());

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], o[i]);
  }

  // Range case, binary op, init, non-aligned ranges
  for (auto &&x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                      12);
  shp::inclusive_scan(shp::par_unseq, v, o, std::multiplies<>(), 12);

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], o[i]);
  }

  // Iterator case, no binary op or init, perfectly aligned
  for (auto &&x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v.begin(), v.end(), v.begin());

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], v[i]);
  }

  // Iterator case, binary op no init, non-aligned ranges
  for (auto &&x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v.begin(), v.end(), o.begin(),
                      std::plus<>());

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], o[i]);
  }

  // Range case, binary op, init, non-aligned ranges
  for (auto &&x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                      12);
  shp::inclusive_scan(shp::par_unseq, v.begin(), v.end(), o.begin(),
                      std::multiplies<>(), 12);

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], o[i]);
  }
}
