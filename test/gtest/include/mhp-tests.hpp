// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/mhp.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;

namespace zhp = rng::views;

#include "common-tests.hpp"

// MHP specific tests
template <typename T> class MhpTests : public testing::Test {
public:
};
TYPED_TEST_SUITE_P(MhpTests);
