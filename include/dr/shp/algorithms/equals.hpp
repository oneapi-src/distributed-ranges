// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>

#include <oneapi/dpl/async>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/shp/algorithms/execution_policy.hpp>
#include <dr/shp/init.hpp>
#include <sycl/sycl.hpp>
namespace dr::shp {

template <typename ExecutionPolicy, dr::distributed_range R1,
          dr::distributed_range R2>
bool equals(ExecutionPolicy &&policy, R1 &&r1, R2 &&r2) {
  return true;
}

} // namespace dr::shp
