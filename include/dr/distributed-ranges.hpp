#pragma once

#include <cassert>
#include <concepts>
#include <iterator>
#include <span>
#include <vector>

// MPI should be optional
#include "mpi.h"

#ifdef DR_STD_RNG
#include <ranges>
namespace rng = std::ranges;
#else
// clang++/icpx do not work with /usr/include/c++/11/ranges
#include "range/v3/all.hpp"
namespace rng = ranges;
#endif

namespace lib {

#include "concepts/concepts.hpp"

#include "details/distributions.hpp"

#include "details/remote_memory.hpp"

#include "details/remote_vector.hpp"

#include "details/distributed_vector.hpp"

#include "details/remote_span.hpp"

#include "details/distributed_span.hpp"

#include "details/execution_policies.hpp"

#include "algorithms/for_each.hpp"
#include "collectives/copy.hpp"
} // namespace lib
