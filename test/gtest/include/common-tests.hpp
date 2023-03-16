// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

template <lib::distributed_range DR>
using LocalVec = std::vector<typename DR::value_type>;

template <typename T> struct Ops1 {
  Ops1(std::size_t n) : dist_vec(n), vec(n) {
    iota(dist_vec, 100);
    rng::iota(vec, 100);
  }

  T dist_vec;
  LocalVec<T> vec;
};

template <typename T> struct Ops2 {
  Ops2(std::size_t n) : dist_vec0(n), dist_vec1(n), vec0(n), vec1(n) {
    iota(dist_vec0, 100);
    iota(dist_vec1, 200);
    rng::iota(vec0, 100);
    rng::iota(vec1, 200);
  }

  T dist_vec0, dist_vec1;
  LocalVec<T> vec0, vec1;
};

template <typename T> struct Ops3 {
  Ops3(std::size_t n)
      : dist_vec0(n), dist_vec1(n), dist_vec2(n), vec0(n), vec1(n), vec2(n) {
    iota(dist_vec0, 100);
    iota(dist_vec1, 200);
    iota(dist_vec2, 300);
    rng::iota(vec0, 100);
    rng::iota(vec1, 200);
    rng::iota(vec2, 300);
  }

  T dist_vec0, dist_vec1, dist_vec2;
  LocalVec<T> vec0, vec1, vec2;
};

bool is_equal(rng::range auto &&r1, rng::range auto &&r2) {
  if (rng::distance(rng::begin(r1), rng::end(r1)) !=
      rng::distance(rng::begin(r2), rng::end(r2))) {
    return false;
  }
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

auto equal_message(rng::range auto &&ref, rng::range auto &&actual,
                   std::string title = " ") {
  if (is_equal(ref, actual)) {
    return fmt::format("");
  }
  return fmt::format("\n{}"
                     "    ref:    {}\n"
                     "    actual: {}\n  ",
                     title == "" ? "" : "    " + title + "\n",
                     rng::views::all(ref), rng::views::all(actual));
}

std::string unary_check_message(rng::range auto &&in, rng::range auto &&ref,
                                rng::range auto &&tst, std::string title = "") {
  if (is_equal(ref, tst)) {
    return "";
  } else {
    return fmt::format("\n{}"
                       "    in:     {}\n"
                       "    ref:    {}\n"
                       "    actual: {}\n  ",
                       title == "" ? "" : "    " + title + "\n", in, ref, tst);
  }
}

std::string check_segments_message(rng::range auto &&r) {
  auto &&segments = lib::ranges::segments(r);
  auto &&flat = rng::join_view(segments);
  if (is_equal(r, flat)) {
    return "";
  }
  return fmt::format("\n"
                     "    Segments does not match distributed range\n"
                     "      range:    {}\n"
                     "      segments: {}\n  ",
                     rng::views::all(r), rng::views::all(segments));
}

auto check_view_message(rng::range auto &&ref, rng::range auto &&actual) {
  return check_segments_message(actual) +
         equal_message(ref, actual, "view mismatch");
}

auto check_mutable_view_message(auto &ops, rng::range auto &&ref,
                                rng::range auto &&actual) {
  // Check view
  auto message = check_view_message(ref, actual);

  barrier();

  // Mutate view
  auto negate = [](auto &&val) { val = -val; };
  auto input_vector = ops.vec;
  std::vector input_view(ref.begin(), ref.end());
  xhp::for_each(default_policy(ops.dist_vec), actual, negate);
  rng::for_each(ref, negate);

  // Check mutated view
  message +=
      unary_check_message(input_view, actual, ref, "mutated view mismatch");

  // Check underlying dv
  message += unary_check_message(input_vector, ops.vec, ops.dist_vec,
                                 "mutated distributed range mismatch");

  return message;
}

auto gtest_result(const auto &message) {
  if (message == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << message;
  }
}

auto equal(rng::range auto &&ref, rng::range auto &&actual,
           std::string title = " ") {
  return gtest_result(equal_message(ref, actual, title));
}

auto check_unary_op(rng::range auto &&in, rng::range auto &&ref,
                    rng::range auto &&tst, std::string title = "") {
  return gtest_result(unary_check_message(in, ref, tst, title));
}

auto check_binary_check_op(rng::range auto &&a, rng::range auto &&b,
                           rng::range auto &&ref, rng::range auto &&actual) {
  if (is_equal(ref, actual)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure()
           << fmt::format("\n        a: {}\n        b: {}\n      ref: {}\n    "
                          "actual: {}\n  ",
                          a, b, ref, actual);
  }
}

auto check_segments(std::forward_iterator auto di) {
  auto &&segments = lib::ranges::segments(di);
  auto &&flat = rng::join_view(segments);
  if (is_equal(di, flat)) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
         << fmt::format("\n    segments: {}\n  ", segments);
}

auto check_segments(rng::range auto &&dr) {
  return gtest_result(check_segments_message(dr));
}

auto check_view(rng::range auto &&ref, rng::range auto &&actual) {
  return gtest_result(check_view_message(ref, actual));
}

auto check_mutable_view(auto &op, rng::range auto &&ref,
                        rng::range auto &&actual) {
  return gtest_result(check_mutable_view_message(op, ref, actual));
}

template <typename T>
std::vector<T> generate_random(std::size_t n, std::size_t bound = 25) {
  std::vector<T> v;
  v.reserve(n);

  for (std::size_t i = 0; i < n; i++) {
    auto r = lrand48() % bound;
    v.push_back(r);
  }

  return v;
}
