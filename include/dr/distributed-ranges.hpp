#pragma once

#include <cassert>
#include <concepts>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <vector>

#ifdef DR_FORMAT
#include <fmt/core.h>
#include <fmt/ranges.h>
#endif

#include <source_location/source_location.hpp>

// MPI should be optional
#include "mkl.h"
#include "mpi.h"

#ifdef DR_STD_RNG
#include <ranges>
namespace rng = std::ranges;
#else
// clang++/icpx do not work with /usr/include/c++/11/ranges
#include "range/v3/all.hpp"
namespace rng = ranges;
#endif

#include <experimental/mdarray>
#include <experimental/mdspan>
namespace stdex = std::experimental;

#include "details/logger.hpp"

#include "details/allocators.hpp"

#include "details/common.hpp"
#include "details/communicator.hpp"

#include "concepts/concepts.hpp"

#include "details/distributions.hpp"

#include "details/remote_memory.hpp"

#include "details/remote_vector.hpp"

#include "details/halo.hpp"

#include "details/distributed_vector.hpp"

#include "details/distributed_mdspan.hpp"

#include "details/remote_span.hpp"

#include "details/distributed_span.hpp"

#include "details/execution_policies.hpp"

#include "algorithms/algorithms.hpp"
#include "algorithms/copy.hpp"
#include "algorithms/for_each.hpp"
#include "algorithms/transpose.hpp"
