// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

template <typename... Rs> auto test_zip(Rs &&...rs) {
  return dr::mhp::views::zip_base(std::forward<Rs>(rs)...);
}

// Fixture
class ZipLocal : public ::testing::Test {
protected:
  void SetUp() override {
    int val = 100;
    for (std::size_t i = 0; i < ops.size(); i++) {
      auto &op = ops[i];
      auto &mop = mops[i];
      op.resize(10);
      mop.resize(10);
      rng::iota(op, val);
      rng::iota(mop, val);
      val += 100;
    }
  }

  std::array<std::vector<int>, 3> ops;
  std::array<std::vector<int>, 3> mops;
};

// Try 1, 2, 3 to check pair/tuple issues
TEST_F(ZipLocal, Op1) { EXPECT_EQ(rng::views::zip(ops[0]), test_zip(ops[0])); }

TEST_F(ZipLocal, Op2) {
  EXPECT_EQ(rng::views::zip(ops[0], ops[1]), test_zip(ops[0], ops[1]));
}

TEST_F(ZipLocal, Op3) {
  EXPECT_EQ(rng::views::zip(ops[0], ops[1], ops[2]),
            test_zip(ops[0], ops[1], ops[2]));
}

TEST_F(ZipLocal, Distance) {
  EXPECT_EQ(rng::distance(rng::views::zip(ops[0], ops[1], ops[2])),
            rng::distance(test_zip(ops[0], ops[1], ops[2])));
}

TEST_F(ZipLocal, Size) {
  auto z = test_zip(ops[0], ops[1]);
  EXPECT_EQ(rng::size(ops[0]), rng::size(z));
}

TEST_F(ZipLocal, Begin) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = rng::views::zip(ops[0], ops[1]);
  EXPECT_EQ(*r.begin(), *z.begin());
}

TEST_F(ZipLocal, End) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = rng::views::zip(ops[0], ops[1]);
  EXPECT_EQ(r.end() - r.begin(), z.end() - z.begin());
}

TEST_F(ZipLocal, IterPlusPlus) {
  auto z = test_zip(ops[0], ops[1]);
  auto r = rng::views::zip(ops[0], ops[1]);
  auto z_iter = z.begin();
  z_iter++;
  auto r_iter = r.begin();
  r_iter++;
  EXPECT_EQ(*r_iter, *z_iter);
}

TEST_F(ZipLocal, IterEquals) {
  auto z = test_zip(ops[0], ops[1]);
  EXPECT_TRUE(z.begin() == z.begin());
  EXPECT_FALSE(z.begin() == z.begin() + 1);
}

TEST_F(ZipLocal, For) {
  auto z = test_zip(ops[0], ops[1]);

  auto i = 0;
  for (auto it = z.begin(); it != z.end(); it++) {
    // values are same
    EXPECT_EQ(this->ops[0][i], std::get<0>(*it));
    EXPECT_EQ(this->ops[1][i], std::get<1>(*it));

    // addresses are same
    EXPECT_EQ(&(this->ops[0][i]), &std::get<0>(*it));
    EXPECT_EQ(&(this->ops[1][i]), &std::get<1>(*it));
    i++;
  };
  EXPECT_EQ(ops[0].size(), i);
}

TEST_F(ZipLocal, RangeFor) {
  auto z = test_zip(ops[0], ops[1]);

  auto i = 0;
  for (auto v : z) {
    // values are same
    EXPECT_EQ(this->ops[0][i], std::get<0>(v));
    EXPECT_EQ(this->ops[1][i], std::get<1>(v));

    // addresses are same
    EXPECT_EQ(&(this->ops[0][i]), &std::get<0>(v));
    EXPECT_EQ(&(this->ops[1][i]), &std::get<1>(v));
    i++;
  };
  EXPECT_EQ(ops[0].size(), i);
}

TEST_F(ZipLocal, ForEach) {
  auto z = test_zip(ops[0], ops[1]);

  auto i = 0;
  auto check = [this, &i](auto v) {
    // values are same
    EXPECT_EQ(this->ops[0][i], std::get<0>(v));
    EXPECT_EQ(this->ops[1][i], std::get<1>(v));

    // addresses are same
    EXPECT_EQ(&(this->ops[0][i]), &std::get<0>(v));
    EXPECT_EQ(&(this->ops[1][i]), &std::get<1>(v));
    i++;
  };
  rng::for_each(z, check);
  EXPECT_EQ(ops[0].size(), i);
}

TEST_F(ZipLocal, Mutate) {
  auto r = rng::views::zip(ops[0], ops[1]);
  auto x = test_zip(mops[0], mops[1]);

  std::get<0>(r[0]) = 99;
  std::get<0>(x[0]) = 99;
  EXPECT_EQ(ops[0], mops[0]);
  EXPECT_EQ(ops[1], mops[1]);
  static_assert(rng::random_access_range<decltype(x)>);
}

TEST_F(ZipLocal, MutateForEach) {
  auto r = rng::views::zip(ops[0], ops[1]);
  auto x = test_zip(mops[0], mops[1]);
  auto n2 = [](auto &&v) {
    std::get<0>(v) += 1;
    std::get<1>(v) = -std::get<1>(v);
  };

  rng::for_each(r, n2);
  rng::for_each(x, n2);
  EXPECT_EQ(ops[0], mops[0]);
  EXPECT_EQ(ops[1], mops[1]);

  static_assert(rng::random_access_range<decltype(x)>);
}

#if 0
// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE(Zip, AllTypes);

TYPED_TEST(Zip, Dist1) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec);
  auto dist = xhp::views::zip(ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist2) {
  Ops2<TypeParam> ops(10);

  auto local = rng::views::zip(ops.vec0, ops.vec1);
  auto dist = xhp::views::zip(ops.dist_vec0, ops.dist_vec1);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Dist3) {
  Ops3<TypeParam> ops(10);

  EXPECT_TRUE(
      check_view(rng::views::zip(ops.vec0, ops.vec1, ops.vec2),
                 xhp::views::zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

TYPED_TEST(Zip, Dist3Distance) {
  Ops3<TypeParam> ops(10);

  EXPECT_EQ(rng::distance(rng::views::zip(ops.vec0, ops.vec1, ops.vec2)),
            rng::distance(
                xhp::views::zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

TYPED_TEST(Zip, Iota) {
  Ops1<TypeParam> ops(10);

  auto local = rng::views::zip(xhp::views::iota(100), ops.vec);
  auto dist = xhp::views::zip(xhp::views::iota(100), ops.dist_vec);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_TRUE(check_view(local, dist));
}

TYPED_TEST(Zip, Iota2nd) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::zip(ops.vec, xhp::views::iota(100)),
                         xhp::views::zip(ops.dist_vec, xhp::views::iota(100))));
}

#if 0
// doesn't compile in mhp
TEST(Zip, ToTransform) {
  Ops2<xhp::distributed_vector<int>> ops(10);

  auto mul = [](auto v) { return std::get<0>(v) * std::get<1>(v); };
  auto local = rng::views::transform(rng::views::zip(ops.vec0, ops.vec1), mul);
  auto dist =
      xhp::views::transform(xhp::views::zip(ops.dist_vec0, ops.dist_vec1), mul);
  static_assert(compliant_view<decltype(dist)>);
  EXPECT_EQ(local, dist);
}
#endif

TYPED_TEST(Zip, All) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(
      rng::views::zip(rng::views::all(ops.vec0), rng::views::all(ops.vec1)),
      xhp::views::zip(rng::views::all(ops.dist_vec0),
                      rng::views::all(ops.dist_vec1))));
}

TYPED_TEST(Zip, L_Value) {
  Ops2<TypeParam> ops(10);

  auto l_value = rng::views::all(ops.vec0);
  auto d_l_value = rng::views::all(ops.dist_vec0);
  EXPECT_TRUE(
      check_view(rng::views::zip(l_value, rng::views::all(ops.vec1)),
                 xhp::views::zip(d_l_value, rng::views::all(ops.dist_vec1))));
}

TYPED_TEST(Zip, Subrange) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(
      rng::views::zip(rng::subrange(ops.vec0.begin() + 1, ops.vec0.end() - 1),
                      rng::subrange(ops.vec1.begin() + 1, ops.vec1.end() - 1)),
      xhp::views::zip(
          rng::subrange(ops.dist_vec0.begin() + 1, ops.dist_vec0.end() - 1),
          rng::subrange(ops.dist_vec1.begin() + 1, ops.dist_vec1.end() - 1))));
}

TYPED_TEST(Zip, ForEach) {
  Ops2<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(xhp::views::zip(ops.dist_vec0, ops.dist_vec1), copy);
  rng::for_each(rng::views::zip(ops.vec0, ops.vec1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}

TYPED_TEST(Zip, ForEachIota) {
  Ops1<TypeParam> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(xhp::views::zip(xhp::views::iota(100), ops.dist_vec), copy);
  rng::for_each(rng::views::zip(xhp::views::iota(100), ops.vec), copy);

  EXPECT_EQ(ops.vec, ops.dist_vec);
  EXPECT_EQ(ops.vec, ops.dist_vec);
}
#endif
