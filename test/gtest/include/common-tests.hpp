// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "cxxopts.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>

bool is_equal(rng::range auto &&r1, rng::range auto &&r2) {
  for (auto e : rng::zip_view(r1, r2)) {
    if (e.first != e.second) {
      return false;
    }
  }

  return true;
}

bool is_equal(std::forward_iterator auto it, rng::range auto &&r) {
  for (auto e : r) {
    if (*it++ != e) {
      return false;
    }
  }
  return true;
}

testing::AssertionResult check_segments(std::forward_iterator auto di) {
  auto &&segments = lib::ranges::segments(di);
  auto &&flat = rng::join_view(segments);
  if (is_equal(di, flat)) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
         << fmt::format("\n    segments: {}\n  ", segments);
}

testing::AssertionResult check_segments(rng::range auto &&r) {
  auto &&segments = lib::ranges::segments(r);
  auto &&flat = rng::join_view(segments);
  if (is_equal(r, flat)) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
         << fmt::format("\n    range: {}\n    segments: {}\n  ",
                        rng::views::all(r), rng::views::all(segments));
}

testing::AssertionResult equal(rng::range auto &&r1, rng::range auto &&r2) {
  if (is_equal(r1, r2)) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure() << fmt::format(
             "\n    {}\n    {}\n  ", rng::views::all(r1), rng::views::all(r2));
}

testing::AssertionResult unary_check(rng::range auto &&in,
                                     rng::range auto &&ref,
                                     rng::range auto &&tst) {
  if (is_equal(ref, tst)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << fmt::format(
               "\n     in: {}\n    ref: {}\n    tst: {}\n  ", in, ref, tst);
  }
}

testing::AssertionResult binary_check(rng::range auto &&a, rng::range auto &&b,
                                      rng::range auto &&ref,
                                      rng::range auto &&tst) {
  if (is_equal(ref, tst)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << fmt::format(
               "\n      a: {}\n      b: {}\n    ref: {}\n    tst: {}\n  ", a, b,
               ref, tst);
  }
}
