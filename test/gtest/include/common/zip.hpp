// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE(Zip, AllTypes);

TYPED_TEST(Zip, Basic) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(rng::views::zip(ops.vec0, ops.vec1),
                         zhp::views::zip(ops.dist_vec0, ops.dist_vec1)));
}

TYPED_TEST(Zip, All) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(
      rng::views::zip(rng::views::all(ops.vec0), rng::views::all(ops.vec1)),
      zhp::views::zip(rng::views::all(ops.dist_vec0),
                      rng::views::all(ops.dist_vec1))));
}

TYPED_TEST(Zip, Zip3) {
  Ops3<TypeParam> ops(10);

  EXPECT_TRUE(
      check_view(rng::views::zip(ops.vec0, ops.vec1, ops.vec2),
                 zhp::views::zip(ops.dist_vec0, ops.dist_vec1, ops.dist_vec2)));
}

auto zip_inner(auto &&r1, auto &&r2) {
  return zhp::views::zip(rng::subrange(r1.begin() + 1, r1.end() - 1),
                         rng::subrange(r2.begin() + 1, r2.end() - 1));
}

TYPED_TEST(Zip, Subrange) {
  Ops2<TypeParam> ops(10);

  EXPECT_TRUE(check_view(zip_inner(ops.vec0, ops.vec1),
                         zip_inner(ops.dist_vec0, ops.dist_vec1)));
}

// Not AllTypes because SYCL MHP is broken
TEST(ZipExSycl, ForEach) {
  Ops2<xhp::distributed_vector<int>> ops(10);

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };
  xhp::for_each(default_policy(ops.dist_vec0),
                zhp::views::zip(ops.dist_vec0, ops.dist_vec1), copy);
  rng::for_each(rng::views::zip(ops.vec0, ops.vec1), copy);

  EXPECT_EQ(ops.vec0, ops.dist_vec0);
  EXPECT_EQ(ops.vec1, ops.dist_vec1);
}
