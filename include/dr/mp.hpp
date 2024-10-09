// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef SYCL_LANGUAGE_VERSION
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/execution>
#endif

#ifdef DRISHMEM
#include <ishmem.h>
#endif

#include <cassert>
#include <concepts>
#include <execution>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vendor/source_location/source_location.hpp>

// MPI should be optional
#include <mkl.h>
#include <mpi.h>

#include <dr/detail/format_shim.hpp>
#include <dr/detail/ranges_shim.hpp>

// clang-format off
#include <dr/concepts/concepts.hpp>

#include <dr/detail/logger.hpp>
#include <dr/detail/view_detectors.hpp>
#include <dr/detail/segments_tools.hpp>
#include <dr/detail/iterator_adaptor.hpp>
#include <dr/detail/normal_distributed_iterator.hpp>
#include <dr/detail/memory.hpp>
#include <dr/detail/communicator.hpp>
#include <dr/detail/index.hpp>

#include <dr/views/views.hpp>
#include <dr/views/transform.hpp>

#include <dr/mp/halo.hpp>
#include <dr/mp/global.hpp>
#include <dr/mp/sycl_support.hpp>
#include <dr/mp/common_support.hpp>
#include <dr/mp/alignment.hpp>
#include <dr/mp/allocator.hpp>
#include <dr/mp/views/views.hpp>
#include <dr/mp/views/zip.hpp>
#include <dr/mp/views/enumerate.hpp>
#include <dr/mp/views/sliding.hpp>
#include <dr/mp/views/mdspan_view.hpp>
#include <dr/mp/views/submdspan_view.hpp>
#include <dr/mp/algorithms/copy.hpp>
#include <dr/mp/algorithms/count.hpp>
#include <dr/mp/algorithms/equal.hpp>
#include <dr/mp/algorithms/fill.hpp>
#include <dr/mp/algorithms/for_each.hpp>
#include <dr/mp/algorithms/exclusive_scan.hpp>
#include <dr/mp/algorithms/inclusive_scan.hpp>
#include <dr/mp/algorithms/iota.hpp>
#include <dr/mp/algorithms/reduce.hpp>
#include <dr/mp/algorithms/sort.hpp>
#include <dr/mp/algorithms/md_for_each.hpp>
#include <dr/mp/algorithms/transform.hpp>
#include <dr/mp/algorithms/transpose.hpp>
#include <dr/mp/containers/distributed_vector.hpp>
#include <dr/mp/containers/distributed_mdarray.hpp>
