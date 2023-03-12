// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

template <lib::distributed_range DR>
using LocalVec = std::vector<typename DR::value_type>;

template <typename T> struct Op1 {
  Op1(std::size_t n) : dv(n), v(n) {
    iota(dv, 100);
    rng::iota(v, 100);
  }

  T dv;
  LocalVec<T> v;
};

template <typename T> struct Op2 {
  Op2(std::size_t n) : dv_a(n), dv_b(n), v_a(n), v_b(n) {
    iota(dv_a, 100);
    iota(dv_b, 200);
    rng::iota(v_a, 100);
    rng::iota(v_b, 200);
  }

  T dv_a, dv_b;
  LocalVec<T> v_a, v_b;
};

template <typename T> struct Op3 {
  Op3(std::size_t n) : dv_a(n), dv_b(n), dv_c(n), v_a(n), v_b(n), v_c(n) {
    iota(dv_a, 100);
    iota(dv_b, 200);
    iota(dv_c, 200);
    rng::iota(v_a, 100);
    rng::iota(v_b, 200);
    rng::iota(v_c, 200);
  }

  T dv_a, dv_b, dv_c;
  LocalVec<T> v_a, v_b, v_c;
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

auto check_mutable_view_message(auto &op, rng::range auto &&ref,
                                rng::range auto &&actual) {
  // Check view
  auto message = check_view_message(ref, actual);

  barrier();

  // Mutate view
  auto negate = [](auto &val) { val = -val; };
  auto input_vector = op.v;
  std::vector input_view(ref.begin(), ref.end());
  xhp::for_each(default_policy(op.dv), actual, negate);
  rng::for_each(ref, negate);

  // Check mutated view
  message +=
      unary_check_message(input_view, actual, ref, "mutated view mismatch");

  // Check underlying dv
  message += unary_check_message(input_vector, op.v, op.dv,
                                 "mutated distributed range mismatch");

  return message;
}

auto equal(rng::range auto &&ref, rng::range auto &&actual,
           std::string title = " ") {
  auto message = equal_message(ref, actual, title);
  if (message == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << message;
  }
}

auto unary_check(rng::range auto &&in, rng::range auto &&ref,
                 rng::range auto &&tst, std::string title = "") {
  auto result = unary_check_message(in, ref, tst, title);
  if (result == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << result;
  }
}

auto binary_check(rng::range auto &&a, rng::range auto &&b,
                  rng::range auto &&ref, rng::range auto &&tst) {
  if (is_equal(ref, tst)) {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << fmt::format(
               "\n      a: {}\n      b: {}\n    ref: {}\n    tst: {}\n  ", a, b,
               ref, tst);
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
  auto message = check_segments_message(dr);
  if (message == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << message;
    ;
  }
}

auto check_view(rng::range auto &&ref, rng::range auto &&actual) {
  auto message = check_view_message(ref, actual);
  if (message == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << message;
    ;
  }
}

auto check_mutable_view(auto &op, rng::range auto &&ref,
                        rng::range auto &&actual) {
  auto message = check_mutable_view_message(op, ref, actual);
  if (message == "") {
    return testing::AssertionSuccess();
  } else {
    return testing::AssertionFailure() << message;
    ;
  }
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
