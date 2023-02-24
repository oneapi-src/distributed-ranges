// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// clang-format off
#include "../details/ranges_shim.hpp"
#include "../details/segments_tools.hpp"
#include "../details/iterator_adaptor.hpp"
#include "../details/normal_distributed_iterator.hpp"
#include "../views/transform.hpp"
#include "device_span.hpp"
#include "algorithms/execution_policy.hpp"
#include "init.hpp"
#include "algorithms/for_each.hpp"
#include "algorithms/reduce.hpp"
#include "allocators.hpp"
#include "copy.hpp"
#include "containers/dense_matrix.hpp"
#include "containers/sparse_matrix.hpp"
#include "distributed_span.hpp"
#include "distributed_vector.hpp"
#include "range.hpp"
#include "range_adaptors.hpp"
#include "util.hpp"
