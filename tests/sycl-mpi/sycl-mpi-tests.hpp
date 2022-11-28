#include <CL/sycl.hpp>

#include "gtest/gtest.h"

#include "mpi.h"

#include "cxxopts.hpp"

#include "dr/distributed-ranges.hpp"

extern MPI_Comm comm;
extern int comm_rank;
extern int comm_size;
