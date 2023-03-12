// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE(Zip, TestTypes);

TYPED_TEST(Zip, Basic) {
  Op2<TypeParam> ops(10);

  auto d_z = zhp::zip(ops.dv_a, ops.dv_b);
  EXPECT_TRUE(check_segments(d_z));
  EXPECT_TRUE(equal(rng::views::zip(ops.v_a, ops.v_b), d_z));
}

TYPED_TEST(Zip, Zip3) {
  Op3<TypeParam> ops(10);

  auto d_z = zhp::zip(ops.dv_a, ops.dv_b, ops.dv_c);
  EXPECT_TRUE(check_segments(d_z));
  EXPECT_TRUE(equal(rng::views::zip(ops.v_a, ops.v_b, ops.v_c), d_z));
}

auto zip_inner(auto &&r1, auto &&r2) {
  return rng::views::zip(rng::subrange(r1.begin() + 1, r1.end() - 1),
                         rng::subrange(r2.begin() + 1, r2.end() - 1));
}

TYPED_TEST(Zip, Subrange) {
  Op2<TypeParam> ops(10);

  auto dv_z = zip_inner(ops.dv_a, ops.dv_b);
  EXPECT_TRUE(check_segments(dv_z));
  EXPECT_TRUE(equal(zip_inner(ops.v_a, ops.v_b), dv_z));
}
