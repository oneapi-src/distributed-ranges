#pragma once

#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>

namespace lib {

#include "concepts/concepts.hpp"

#include "details/distributions.hpp"
#include "details/remote_memory.hpp"

#include "details/distributed_vector.hpp"

#include "details/remote_span.hpp"

#include "details/distributed_span.hpp"
#include "details/execution_policies.hpp"

#include "algorithms/for_each.hpp"
#include "collectives/copy.hpp"
} // namespace lib
