// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mhp-tests.hpp"

using T = int;
using V = std::vector<T>;
using DV = mhp::distributed_vector<T>;
using DVI = mhp::distributed_vector_iterator<T>;

// Base case. Anything conforms with itself.
template <typename It> auto conformant(It &&iter) {
  return std::pair(true, iter);
}

// Recursive case. This iterator conforms with the rest.
template <lib::distributed_iterator It, typename... Its>
auto conformant(It &&iter, Its &&...iters) {
  auto &&[rest_conforms, constraining_iter] =
      conformant(std::forward<Its>(iters)...);
  return std::pair(rest_conforms && iter.conforms(constraining_iter), iter);
}

// Recursive case. This iterator is non-constraining
template <typename It, typename... Its>
auto conformant(It &&iter, Its &&...iters) {
  return conformant(std::forward<Its>(iters)...);
}

TEST(MhpTests, IteratorConformance) {
  DV dv1(10), dv2(10);
  V v1(10);

  // 2 distributed vectors
  EXPECT_TRUE(conformant(dv1.begin(), dv2.begin()).first);
  ;
  // misaligned distributed vector
  EXPECT_FALSE(conformant(dv1.begin() + 1, dv2.begin()).first);

  // iota conformant with anything
  // EXPECT_TRUE(conformant(dv1.begin(), rng::views::iota(1)).first);
  EXPECT_TRUE(conformant(rng::views::iota(1), dv1.begin()).first);

  // May not be useful to support
  // distributed and local vector
  // EXPECT_FALSE(conformant(dv1.begin(), v1.begin()));
}
