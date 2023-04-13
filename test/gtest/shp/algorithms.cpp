// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "shp-tests.hpp"

using T = int;
using DV = dr::shp::distributed_vector<T, dr::shp::device_allocator<T>>;
using V = std::vector<T>;

// hard to reproduce fails
TEST(ShpTests, InclusiveScan_aligned) {
  std::size_t n = 100;

  // With execution Policy
  {
    dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(n);
    std::vector<int> lv(n);

    // Range case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(dr::shp::par_unseq, v, v);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }

    // Iterator case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(dr::shp::par_unseq, v.begin(), v.end(), v.begin());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }
  }

  // Without execution policies
  {
    dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(n);
    std::vector<int> lv(n);

    // Range case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(v, v);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }

    // Iterator case, no binary op or init, perfectly aligned
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(v.begin(), v.end(), v.begin());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], v[i]);
    }
  }
}

TEST(ShpTests, DISABLED_InclusiveScan_nonaligned) {
  std::size_t n = 100;

  // With execution policies
  {
    dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(n);
    dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> o(
        v.size() * 2);
    std::vector<int> lv(n);

    // Range case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(dr::shp::par_unseq, v, o, std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    dr::shp::inclusive_scan(dr::shp::par_unseq, v, o, std::multiplies<>(), 12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Iterator case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(dr::shp::par_unseq, v.begin(), v.end(), o.begin(),
                            std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    dr::shp::inclusive_scan(dr::shp::par_unseq, v.begin(), v.end(), o.begin(),
                            std::multiplies<>(), 12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }
  }

  // Without execution policies
  {
    dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> v(n);
    dr::shp::distributed_vector<int, dr::shp::device_allocator<int>> o(
        v.size() * 2);
    std::vector<int> lv(n);

    // Range case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(v, o, std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    dr::shp::inclusive_scan(v, o, std::multiplies<>(), 12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Iterator case, binary op no init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
    dr::shp::inclusive_scan(v.begin(), v.end(), o.begin(), std::plus<>());

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }

    // Range case, binary op, init, non-aligned ranges
    for (auto &&x : lv) {
      x = lrand48() % 100;
    }
    dr::shp::copy(lv.begin(), lv.end(), v.begin());

    std::inclusive_scan(lv.begin(), lv.end(), lv.begin(), std::multiplies<>(),
                        12);
    dr::shp::inclusive_scan(v.begin(), v.end(), o.begin(), std::multiplies<>(),
                            12);

    for (std::size_t i = 0; i < lv.size(); i++) {
      EXPECT_EQ(lv[i], o[i]);
    }
  }
}
