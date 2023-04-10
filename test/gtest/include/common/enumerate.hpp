// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Enumerate : public testing::Test {
public:
};

TYPED_TEST_SUITE(Enumerate, AllTypes);

TYPED_TEST(Enumerate, Basic) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_view(shp::views::enumerate(ops.vec),
                         shp::views::enumerate(ops.dist_vec)));
}

TYPED_TEST(Enumerate, Mutate) {
  Ops1<TypeParam> ops(10);

  EXPECT_TRUE(check_mutate_enumerateview(ops, shp::views::enumerate(ops.vec),
                                         shp::views::enumerate(ops.dist_vec)));
}
