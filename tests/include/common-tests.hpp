// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "cxxopts.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

bool is_equal(const rng::range auto &r1, const rng::range auto &r2) {
  // std::ranges::views::zip handles this better, but requires range-v3
  for (std::size_t i = 0;
       r1.begin() + i != r1.end() && r2.begin() + i != r2.end(); i++) {
    if (r1[i] != r2[i]) {
      return false;
    }
  }

  return true;
}

testing::AssertionResult equal(const rng::range auto &r1,
                               const rng::range auto &r2) {
  if (is_equal(r1, r2)) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
         << fmt::format("\n    {}\n    {}\n  ", r1, r2);
}

testing::AssertionResult unary_check(const rng::range auto &in,
                                     const rng::range auto &ref,
                                     const rng::range auto &tst) {
  if (is_equal(ref, tst)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << fmt::format(
               "\n     in: {}\n    ref: {}\n    tst: {}\n  ", in, ref, tst);
  }
}

testing::AssertionResult binary_check(const rng::range auto &a,
                                      const rng::range auto &b,
                                      const rng::range auto &ref,
                                      const rng::range auto &tst) {
  if (is_equal(ref, tst)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << fmt::format(
               "\n      a: {}\n      b: {}\n    ref: {}\n    tst: {}\n  ", a, b,
               ref, tst);
  }
}
