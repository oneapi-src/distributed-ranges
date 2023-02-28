// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "mpi.h"

#include "dr/mhp.hpp"

namespace xhp = mhp;

#include "common-tests.hpp"

template <typename T> class MhpTests : public testing::Test {
public:
};

TYPED_TEST_SUITE_P(MhpTests);

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;
