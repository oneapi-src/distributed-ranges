// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

#ifdef SYCL_LANGUAGE_VERSION

using T = float;

TEST(SYCLUtils, ParalelFor1D) {
  const std::size_t size = 10;
  sycl::queue q;
  sycl::range range(size - 1);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [a](auto i) { a[i] = i; };
  auto setb = [b](auto i) { b[i] = i; };
  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  EXPECT_EQ(rng::span(a, size), rng::span(b, size));
}

void set(auto col_size, auto base, auto index) {
  base[(index[0] + 1) * col_size + index[1] + 1] = 22;
}

TEST(SYCLUtils, ParalelFor2D) {
  const std::size_t row_size = 5, col_size = row_size,
                    size = row_size * col_size;
  sycl::queue q;
  sycl::range range(row_size - 2, col_size - 2);

  auto a = sycl::malloc_shared<T>(size, q);
  auto b = sycl::malloc_shared<T>(size, q);
  std::fill(a, a + size, 99);
  std::fill(b, b + size, 99);
  auto seta = [col_size, base = a](auto index) { set(col_size, base, index); };
  auto setb = [col_size, base = b](auto index) { set(col_size, base, index); };

  q.parallel_for(range, seta).wait();
  dr::__detail::parallel_for(q, range, setb).wait();

  EXPECT_EQ(rng::span(a, size), rng::span(b, size))
      << fmt::format("a:\n{}b:\n{}", md::mdspan(a, row_size, col_size),
                     md::mdspan(b, row_size, col_size));
}

#endif // SYCL_LANGUAGE_VERSION
