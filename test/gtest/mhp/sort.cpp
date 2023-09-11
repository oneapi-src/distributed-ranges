// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"
#include <dr/detail/communicator.hpp>
#include <dr/mhp/algorithms/sort.hpp>

using T = int;
using DV = dr::mhp::distributed_vector<T, dr::mhp::default_allocator<T>>;
using LV = std::vector<T>;

void test_sort(LV &v, auto func) {
  auto size = v.size();
  DV d_v(size);

  for (std::size_t idx = 0; idx < size; idx++) {
    d_v[idx] = v[idx];
  }
  barrier();

  std::sort(v.begin(), v.end(), func);
  dr::mhp::sort(d_v, func);

  barrier();
  EXPECT_TRUE(equal(v, d_v));
}

void test_sort2s(LV &v) {
  test_sort(v, std::less<T>());
  test_sort(v, std::greater<T>());
}

void test_sort_randomvec(std::size_t size, std::size_t bound = 100) {
  LV l_v = generate_random<T>(size, bound);
  test_sort2s(l_v);
}

TEST(MhpSort, Random) {
  test_sort_randomvec(1);
  test_sort_randomvec(comm_size - 1);
  test_sort_randomvec((comm_size - 1) * (comm_size - 1));
  test_sort_randomvec(17);
  test_sort_randomvec(123);
}

TEST(MhpSort, BigRandom) { test_sort_randomvec(1234567, 65535); }

TEST(MhpSort, NonRandom) {
  LV v;
  v = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  test_sort2s(v);

  v = {1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1};
  test_sort2s(v);

  v = {1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 9, 1};
  test_sort2s(v);

  v = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
  test_sort2s(v);

  v = {6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6};
  test_sort2s(v);

  v = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
  test_sort2s(v);
}

TEST(MhpSort, LongSorted) {
  LV v(100000);
  rng::iota(v, 1);
  test_sort2s(v);

  rng::reverse(v);
  test_sort2s(v);
}
