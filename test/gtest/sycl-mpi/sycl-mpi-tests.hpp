// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>

#include "mpi.h"

#include "dr/distributed-ranges.hpp"

#include "common-tests.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;
