#pragma once

#include <cassert>
#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>

#ifdef MPI_VERSION
#include "mpi-utils.hpp"
#endif

#include "data-utils.hpp"
