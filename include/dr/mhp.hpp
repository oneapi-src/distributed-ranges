// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#endif

#include <cassert>
#include <concepts>
#include <execution>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>
// Workaround for doxygen warning about internal inconsistency
namespace fmt {}

#include "vendor/source_location/source_location.hpp"

// MPI should be optional
#include "mkl.h"
#include "mpi.h"

#include "range/v3/all.hpp"
namespace rng = ranges;

// clang-format off
#include "concepts/concepts.hpp"

#include "details/logger.hpp"
#include "details/view_detectors.hpp"
#include "details/segments_tools.hpp"
#include "details/iterator_adaptor.hpp"
#include "details/normal_distributed_iterator.hpp"
#include "details/memory.hpp"
#include "details/communicator.hpp"
#include "details/halo.hpp"

#include "views/views.hpp"
#include "views/transform.hpp"

#include "mhp/global.hpp"
#ifdef SYCL_LANGUAGE_VERSION
#include "mhp/sycl_support.hpp"
#endif
#include "mhp/containers/distributed_vector.hpp"
#include "mhp/alignment.hpp"
#include "mhp/views.hpp"
#include "mhp/algorithms/cpu_algorithms.hpp"
