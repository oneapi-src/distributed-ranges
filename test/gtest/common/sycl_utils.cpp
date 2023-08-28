// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#ifdef SYCL_LANGUAGE_VERSION

using T = float;

TEST(SYCLUtils, ParalelFor1D) {
  const std::size_t size = 10;
  sycl::queue q;
  sycl::range<1> range(size - 1);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [a](auto i) { a[i] = i; };
  auto setb = [b](auto i) { b[i] = i; };
  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  EXPECT_NE(rng::span(a, size), rng::span(a, size));
}

#endif // SYCL_LANGUAGE_VERSION
