// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <span>
#include <sycl/sycl.hpp>
#include <vector>

namespace shp {

struct device_policy {
  // for now, empty as we always use all devices returned by shp::devices()
};

} // namespace shp
