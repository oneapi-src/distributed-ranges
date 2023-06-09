// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "xhp-tests.hpp"

// Fixture
template <typename T> class InclusiveScan : public testing::Test {
public:
};

TYPED_TEST_SUITE(InclusiveScan, AllTypes);

TYPED_TEST(InclusiveScan, Basic) {
    TypeParam dv(10) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TypeParam dv_expected_result(10) = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
    xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin());
    EXPECT_EQ(dv, dv_expected_result);
}

TYPED_TEST(InclusiveScan, BinaryOperationPlus) {
    TypeParam dv(10) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    TypeParam dv_expected_result(10) = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
    xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin(), std::plus<>());
    EXPECT_EQ(dv, dv_expected_result);
}

TYPED_TEST(InclusiveScan, BinaryOperationMinus) {
    TypeParam dv(10) = {45, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    TypeParam dv_expected_result(10) = {45, 44, 42, 39, 35, 30, 24, 17, 9, 0};
    xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin());
    EXPECT_EQ(dv, dv_expected_result);
}

TYPED_TEST(InclusiveScan, BinaryOperationMultiplies) {
    TypeParam dv(5) = {1, 2, 3, 4, 5};
    TypeParam dv_expected_result(5) = {1, 2, 6, 24, 120};
    xhp::inclusive_scan(dv.begin(), dv.end(), dv.begin());
    EXPECT_EQ(dv, dv_expected_result);
}


// TYPED_TEST(IncScan, mutate) {
//   Ops1<TypeParam> ops(10);

//   EXPECT_TRUE(check_mutate_view(ops, rng::views::IncScan(ops.vec, 6),
//                                 xhp::views::IncScan(ops.dist_vec, 6)));
// }

// template <class TypeParam>
// void localAndDrIncScanResultsAreSameTest(std::size_t IncScanSize) {
//   Ops1<TypeParam> ops(10);
//   auto dist = xhp::views::IncScan(ops.dist_vec, IncScanSize);
//   auto local = rng::views::IncScan(ops.vec, IncScanSize);
//   EXPECT_TRUE(check_view(local, dist));
// }

// TYPED_TEST(IncScan, lessThanSize) {
//   localAndDrIncScanResultsAreSameTest<TypeParam>(6);
// }

// TYPED_TEST(IncScan, sameSize) { localAndDrIncScanResultsAreSameTest<TypeParam>(10); }

// TYPED_TEST(IncScan, moreSize) { localAndDrIncScanResultsAreSameTest<TypeParam>(12); }

// TYPED_TEST(IncScan, zero) { localAndDrIncScanResultsAreSameTest<TypeParam>(0); }

// TYPED_TEST(IncScan, one) { localAndDrIncScanResultsAreSameTest<TypeParam>(1); }

// TYPED_TEST(IncScan, emptyInput_zeroSize) {
//   TypeParam dv(0);
//   auto dist = xhp::views::IncScan(dv, 0);
//   EXPECT_TRUE(rng::empty(dist));
// }

// TYPED_TEST(IncScan, emptyInput_nonZeroSize) {
//   TypeParam dv(0);
//   auto dist = xhp::views::IncScan(dv, 1);
//   EXPECT_TRUE(rng::empty(dist));
// }

// TYPED_TEST(IncScan, large) {
//   TypeParam dv(123456, 77);

//   auto IncScan_result = xhp::views::IncScan(dv, 54321);

//   EXPECT_EQ(*(--IncScan_result.end()), 77);
//   fence();
//   *(--IncScan_result.end()) = 5;
//   fence();
//   EXPECT_EQ(dv[54320], 5);
//   EXPECT_EQ(dv[54321], 77);
//   EXPECT_EQ(rng::size(IncScan_result), 54321);
// }

// TYPED_TEST(IncScan, IncScanOfOneElementHasOneSegmentAndSameRank) {
//   TypeParam dv(10, 77);
//   auto IncScan_view_result = xhp::views::IncScan(dv, 1);

//   auto IncScan_view_segments = dr::ranges::segments(IncScan_view_result);
//   auto dv_segments = dr::ranges::segments(dv);

//   EXPECT_TRUE(check_segments(IncScan_view_result));
//   EXPECT_EQ(rng::size(IncScan_view_segments), 1);
//   EXPECT_EQ(dr::ranges::rank(IncScan_view_segments[0]),
//             dr::ranges::rank(dv_segments[0]));
// }

// TYPED_TEST(IncScan, IncScanOfFirstSegementHasOneSegmentAndSameRank) {
//   TypeParam dv(10, 77);

//   const auto first_seg_size = dr::ranges::segments(dv)[0].size();
//   auto IncScan_view_result = xhp::views::IncScan(dv, first_seg_size);
//   auto IncScan_view_segments = dr::ranges::segments(IncScan_view_result);
//   EXPECT_EQ(rng::size(IncScan_view_segments), 1);
//   EXPECT_EQ(dr::ranges::rank(IncScan_view_segments[0]),
//             dr::ranges::rank(dr::ranges::segments(dv)[0]));
// }

// template <class TypeParam>
// void IncScanHasSameSegments(std::size_t dv_size, std::size_t IncScan_size) {
//   TypeParam dv(dv_size, 77);

//   auto dv_segments = dr::ranges::segments(dv);
//   auto IncScan_view_result = xhp::views::IncScan(dv, IncScan_size);
//   auto IncScan_view_segments = dr::ranges::segments(IncScan_view_result);

//   EXPECT_EQ(rng::size(dv_segments), rng::size(IncScan_view_segments));
//   for (std::size_t i = 0; i < rng::size(dv_segments); ++i)
//     EXPECT_EQ(dr::ranges::rank(dv_segments[i]),
//               dr::ranges::rank(IncScan_view_segments[i]));
// }

// TYPED_TEST(IncScan, IncScanOfAllButOneSizeHasAllSegmentsWithSameRanks) {
//   IncScanHasSameSegments<TypeParam>(EVENLY_DIVIDABLE_SIZE,
//                                  EVENLY_DIVIDABLE_SIZE - 1);
// }

// TYPED_TEST(IncScan, IncScanOfMoreSizeHasSameNumberOfSegmentsAndSameRanks) {
//   IncScanHasSameSegments<TypeParam>(EVENLY_DIVIDABLE_SIZE,
//                                  EVENLY_DIVIDABLE_SIZE * 2);
// }
