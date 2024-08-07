// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xp-tests.hpp"
#include <dr/sp.hpp>

namespace sp = dr::sp;

TEST(DetailTest, parallel_for) {
  std::size_t size = 2 * 1024 * 1024;
  std::size_t n = 4 * std::size_t(std::numeric_limits<int32_t>::max());

  // Compute `v`
  std::vector<int> v(size, 0);

  auto iota = ranges::views::iota(std::size_t(0), n);

  std::for_each(iota.begin(), iota.end(), [&](auto i) { v[i % size] += 1; });

  auto &&q = sp::__detail::queue(0);

  sp::shared_allocator<int> alloc(q);

  sp::vector<int, sp::shared_allocator<int>> dvec(size, 0, alloc);

  auto dv = dvec.data();

  dr::__detail::parallel_for(q, n, [=](auto i) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        v(dv[i % size]);
    v += 1;
  }).wait();

  std::vector<int> dvec_local(size);
  sp::copy(dvec.begin(), dvec.end(), dvec_local.begin());

  EXPECT_EQ(v, dvec_local);
}
