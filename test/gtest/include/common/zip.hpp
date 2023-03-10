// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// Fixture
template <typename T> class Zip : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(Zip);

TYPED_TEST_P(Zip, Basic) {
  std::size_t n = 10;

  TypeParam dv_a(n), dv_b(n), dv_c(n);
  iota(dv_a, 100);
  iota(dv_b, 200);
  iota(dv_c, 300);

  auto d_z2 = zhp::zip(dv_a, dv_b);
  EXPECT_TRUE(check_segments(d_z2));

  auto d_z3 = zhp::zip(dv_a, dv_b, dv_c);
  EXPECT_TRUE(check_segments(d_z3));
  barrier();

  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n), b(n), c(n);
    rng::iota(a, 100);
    rng::iota(b, 200);
    rng::iota(c, 300);

    auto z2 = rng::views::zip(a, b);
    EXPECT_TRUE(equal(z2, d_z2));

    auto z3 = rng::views::zip(a, b, c);
    EXPECT_TRUE(equal(z3, d_z3));
  }
}

auto zip_inner(auto &&r1, auto &&r2) {
  return rng::views::zip(rng::subrange(r1.begin() + 1, r1.end() - 1),
                         rng::subrange(r2.begin() + 1, r2.end() - 1));
}

TYPED_TEST_P(Zip, Subrange) {
  std::size_t n = 10;

  TypeParam dv_a(n), dv_b(n);
  iota(dv_a, 100);
  iota(dv_b, 200);

  auto dv_z = zip_inner(dv_a, dv_b);
  EXPECT_TRUE(check_segments(dv_z));
  barrier();

  if (comm_rank == 0) {
    LocalVec<TypeParam> a(n), b(n);
    rng::iota(a, 100);
    rng::iota(b, 200);

    auto z = zip_inner(a, b);
    EXPECT_TRUE(equal(z, dv_z));
  }
}

REGISTER_TYPED_TEST_SUITE_P(Zip, Basic, Subrange);
INSTANTIATE_TYPED_TEST_SUITE_P(MHP, Zip, TestTypes);
