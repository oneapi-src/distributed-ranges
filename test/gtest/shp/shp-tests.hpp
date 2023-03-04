// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sycl/sycl.hpp>

#include "dr/shp/shp.hpp"

extern int comm_rank;
extern int comm_size;

namespace zhp = shp::views;
namespace xhp = shp;

inline void barrier() {}
inline void fence() {}

#include "common-tests.hpp"
