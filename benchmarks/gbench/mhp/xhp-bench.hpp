// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../common/common-bench.hpp"

#include "dr/mhp.hpp"

namespace xhp = dr::mhp;

#define BENCH_MHP

extern std::size_t stencil_steps;
extern std::size_t num_rows;
extern std::size_t num_columns;
extern bool check_results;
