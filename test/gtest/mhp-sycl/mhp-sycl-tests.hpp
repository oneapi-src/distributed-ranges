// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sycl.hpp>

#include "mpi.h"

#include "dr/mhp.hpp"

#include "common-tests.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;
