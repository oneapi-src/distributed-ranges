// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

using T = int;
using DV = xhp::distributed_vector<T>;
using LV = std::vector<T>;

void test_sort(LV v, auto func) {
  auto size = v.size();
  DV d_v(size);

  for (std::size_t idx = 0; idx < size; idx++) {
    d_v[idx] = v[idx];
  }
  barrier();

  std::sort(v.begin(), v.end(), func);
  xhp::sort(d_v, func);

  EXPECT_TRUE(equal(v, d_v));
}

void test_sort2s(LV v) {
  test_sort(v, std::less<T>());
  test_sort(v, std::greater<T>());
}

void test_sort_randomvec(std::size_t size, std::size_t bound = 100) {
  LV l_v = generate_random<T>(size, bound);
  test_sort2s(l_v);
}

TEST(Sort, Random_1) { test_sort_randomvec(1); }

TEST(Sort, Random_CommSize_m1) { test_sort_randomvec(comm_size - 1); }

TEST(Sort, Random_CommSize_m1_sq) {
  test_sort_randomvec((comm_size - 1) * (comm_size - 1));
}

TEST(Sort, Random_dist_small) { test_sort_randomvec(17); }

TEST(Sort, Random_dist_med) { test_sort_randomvec(123); }

TEST(Sort, Special) {
  LV v = {
      101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
      111, 112, 113, 114, 115, 116, 117, 118, 119, 120, //
      201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
      211, 212, 213, 214, 215, 216, 217, 218, 219, 220, //
      301, 302, 303, 304, 305, 306, 307, 308, 309, 310,
      311, 312, 313, 314, 315, 316, 317, 318, 319, 320, //
      401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
      411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
  };
  test_sort(v, std::less<T>());
}

TEST(Sort, Special2) {
  LV v = {
      101, 105, 109, 113, 117, 201, 205, 209, 213, 217,
      301, 305, 309, 313, 317, 401, 405, 409, 413, 417, //
      102, 106, 110, 114, 118, 202, 206, 210, 214, 218,
      302, 306, 310, 314, 318, 402, 406, 410, 414, 418, //
      103, 107, 111, 115, 119, 203, 207, 211, 215, 219,
      303, 307, 311, 315, 319, 403, 407, 411, 415, 419, //
      104, 108, 112, 116, 120, 204, 208, 212, 216, 220,
      304, 308, 312, 316, 320, 404, 408, 412, 416, 420,
  };
  test_sort(v, std::less<T>());
}

TEST(Sort, Special3) {
  LV v = {1,  5,  9,  13, 17, 21, 25, 29, 33, 37, 41, 45, 49,
          53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, //
          2,  6,  10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50,
          54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, //
          3,  7,  11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51,
          55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, //
          4,  8,  12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52,
          56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100};
  test_sort(v, std::less<T>());
}

TEST(Sort, AllSame) {
  test_sort2s({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST(Sort, AllSameButOneMid) {
  test_sort2s({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1});
}

TEST(Sort, AllSameButOneEnd) {
  test_sort2s({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9});
}

TEST(Sort, AllSameButOneSmaller) {
  test_sort2s({5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(Sort, AllSameButOneBigger) {
  test_sort2s({5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(Sort, AllSameButOneSBeg) {
  test_sort2s({5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}
TEST(Sort, AllSameButOneBBeg) {
  test_sort2s({5, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(Sort, MostSame) { test_sort2s({1, 9, 2, 2, 2, 2, 2, 2, 2, 2, 9, 1}); }

TEST(Sort, Pyramid) { test_sort2s({1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1}); }

TEST(Sort, RevPyramid) { test_sort2s({6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6}); }

TEST(Sort, Wave) { test_sort2s({1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}); }

TEST(Sort, LongSorted) {
  LV v(100000);
  rng::iota(v, 1);
  test_sort2s(v);

  rng::reverse(v);
  test_sort2s(v);
}
