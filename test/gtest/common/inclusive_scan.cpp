// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class InclusiveScan : public testing::Test {
public:
};

TYPED_TEST_SUITE(InclusiveScan, AllTypes);

// TYPED_TEST(InclusiveScan, Basic) {
//     TypeParam dv(5);
//     xhp::iota(dv, 1);
//     TypeParam dv_expected_result(5) = {1, 3, 6, 10, 15};
//     xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin());
//     EXPECT_EQ(dv, dv_expected_result);
// }

// TYPED_TEST(InclusiveScan, BinaryOperationPlus) {
//     TypeParam dv(5);
//     xhp::iota(dv, 1);
//     TypeParam dv_expected_result(5) = {1, 3, 6, 10, 15};
//     xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin(), std::plus<>());
//     EXPECT_EQ(dv, dv_expected_result);
// }

// TYPED_TEST(InclusiveScan, BinaryOperationMinus) {
//     std::vector<int> v = {}
//     TypeParam dv(10) = {45, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//     TypeParam dv_expected_result(10) = {45, 44, 42, 39, 35, 30, 24, 17, 9, 0};
//     xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin(), std::minus<>());
//     EXPECT_EQ(dv, dv_expected_result);
// }

// TYPED_TEST(InclusiveScan, BinaryOperationMultiplies) {
//     TypeParam dv(5);
//     xhp::iota(dv, 1);
//     TypeParam dv_expected_result(5) = {1, 2, 6, 24, 120};
//     xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin(), std::multiplies<>());
//     EXPECT_EQ(dv, dv_expected_result);
// }


TYPED_TEST(InclusiveScan, InclusiveScanSegments) {
    std::vector<int> v = {1, 3, 6, 10, 15, 21};
    TypeParam dv(10);
    xhp::iota(dv, 1);
    auto seg_num = dr::ranges::segments(dv);
    TypeParam dv_expected_result(6);
    xhp::copy(v.begin(), v.end(), dv_expected_result.begin());
    dr::mhp::inclusive_scan(dv.begin(), dv.end(), dv.begin(), std::plus<>());
    // EXPECT_EQ(dv, dv_expected_result);
    EXPECT_EQ(dv, dv_expected_result);
}

