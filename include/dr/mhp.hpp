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

#include <dr/mhp/halo.hpp>
#include <dr/mhp/global.hpp>
#include <dr/mhp/sycl_support.hpp>
#include <dr/mhp/alignment.hpp>
#include <dr/mhp/views/views.hpp>
#include <dr/mhp/views/zip.hpp>
#include <dr/mhp/views/enumerate.hpp>
#include <dr/mhp/views/sliding.hpp>
#include <dr/mhp/views/mdspan_view.hpp>
#include <dr/mhp/views/submdspan_view.hpp>
#include <dr/mhp/algorithms/copy.hpp>
#include <dr/mhp/algorithms/fill.hpp>
#include <dr/mhp/algorithms/for_each.hpp>
#include <dr/mhp/algorithms/inclusive_scan.hpp>
#include <dr/mhp/algorithms/iota.hpp>
#include <dr/mhp/algorithms/reduce.hpp>
#include <dr/mhp/algorithms/stencil_for_each.hpp>
#include <dr/mhp/algorithms/transform.hpp>
#include <dr/mhp/containers/distributed_vector.hpp>
#include <dr/mhp/containers/distributed_dense_matrix.hpp>
#include <dr/mhp/containers/distributed_mdarray.hpp>
