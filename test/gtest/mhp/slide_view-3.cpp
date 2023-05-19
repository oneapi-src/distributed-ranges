// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

// this test is going to be updated with PR with sliding-view
// however it is commited as empty one in advance because another PR needs to
// create its own MPI 3proc only tests and some exaple was needed

#include "xhp-tests.hpp"

template <typename T> class Slide3 : public testing::Test {};

TYPED_TEST_SUITE(Slide3, AllTypes);

TYPED_TEST(Slide3, suite_works_for_3_processes_only) {
  EXPECT_EQ(dr::mhp::default_comm().size(), 3); // dr-style ignore
}

// more Slide3 tests are comming, all assume that there are 3 mpi processes
