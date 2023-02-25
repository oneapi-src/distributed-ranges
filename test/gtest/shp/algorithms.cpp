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

struct negate {
  void operator()(auto &v) const { v = -v; }
};

TEST(ShpTests, ForEach) {
  std::size_t n = 10;

  V a(n), a_in(n);
  std::iota(a.begin(), a.end(), 100);
  std::iota(a_in.begin(), a_in.end(), 100);
  rng::for_each(a, negate{});

  DV dv_a(n);
  std::iota(dv_a.begin(), dv_a.end(), 100);
  shp::for_each(shp::par_unseq, dv_a, negate{});
  EXPECT_TRUE(unary_check(a_in, a, dv_a));
}

TEST(ShpTests, ReduceBasic) {
  std::size_t n = 10;

  V v(n);
  std::iota(v.begin(), v.end(), 10);
  DV dv(n);
  std::iota(dv.begin(), dv.end(), 10);
  auto dvalue = shp::reduce(shp::par_unseq, dv, int(0), std::plus<>());
  auto value = std::reduce(dv.begin(), dv.end(), int(0), std::plus<>());
  EXPECT_EQ(dvalue, value);

  // Iterator versions
  EXPECT_EQ(dvalue, shp::reduce(shp::par_unseq, dv.begin(), dv.end(), int(0),
                                std::plus<>()));
  EXPECT_EQ(dvalue, shp::reduce(shp::par_unseq, dv.begin(), dv.end(), int(0)));
  EXPECT_EQ(dvalue, shp::reduce(shp::par_unseq, dv.begin(), dv.end()));

  // Simplified range versions
  EXPECT_EQ(dvalue, shp::reduce(shp::par_unseq, dv, int(0)));
  EXPECT_EQ(dvalue, shp::reduce(shp::par_unseq, dv));
}

TEST(ShpTests, InclusiveScan) {
  std::size_t n = 100;

  shp::distributed_vector<int, shp::device_allocator<int>> v(n);
  std::vector<int> lv(n);

  for (auto&& x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v, v);

  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], v[i]);
  }

  for (auto&& x : lv) {
    x = lrand48() % 100;
  }
  shp::copy(lv.begin(), lv.end(), v.begin());

  shp::distributed_vector<int, shp::device_allocator<int>> o(v.size()*2);

  std::inclusive_scan(lv.begin(), lv.end(), lv.begin());
  shp::inclusive_scan(shp::par_unseq, v, o);
  
  for (size_t i = 0; i < lv.size(); i++) {
    EXPECT_EQ(lv[i], o[i]);
  }
}

#if 0
void check_reduce(std::size_t n, std::size_t b, std::size_t e) {
  auto op = std::plus<>();
  int init = 10000;
  int iota_base = 100;
  int root = 0;

  DV dv1(n);
  std::iota(dv1.begin(), dv1.end(), iota_base);
  auto dv1_sum = shp::reduce(shp::par_unseq, std::ranges::subrange(dv1.begin() + b, dv1.begin() + e), init, op);

  DV dv2(n);
  std::iota(dv2.begin(), dv1.end(), iota_base);

  auto dv2_sum = std::reduce(dv2.begin() + b, dv2.begin() + e, init, op);

  V v(n);
  std::iota(v.begin(), v.end(), iota_base);
  auto v_sum = std::reduce(v.begin() + b, v.begin() + e, init, op);
  EXPECT_EQ(v_sum, dv1_sum);
  EXPECT_EQ(v_sum, dv2_sum);
}

TEST(CpuMpiTests, ReduceComplex) {
  std::size_t n = 10;

  check_reduce(n, 0, n);
  check_reduce(n, n / 2 - 1, n / 2 + 1);
}
#endif
