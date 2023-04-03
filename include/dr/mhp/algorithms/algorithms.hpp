// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <execution>
#include <type_traits>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/details/logger.hpp>
#include <dr/details/onedpl_direct_iterator.hpp>
#include <dr/details/ranges_shim.hpp>

namespace mhp {

//
//
// fill
//
//

/// Collective fill on distributed range
void fill(lib::distributed_contiguous_range auto &&dr, auto value) {
  for (const auto &s : local_segments(dr)) {
    rng::fill(s, value);
  }
  barrier();
}

/// Collective fill on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void fill(DI first, DI last, auto value) {
  mhp::fill(rng::subrange(first, last), value);
}

//
//
// copy
//
//

/// Copy
void copy(lib::distributed_contiguous_range auto &&in,
          lib::distributed_iterator auto out) {
  if (aligned(in, out)) {
    lib::drlog.debug("copy: parallel execution\n");
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::copy(in_seg, rng::begin(out_seg));
    }
    barrier();
  } else {
    lib::drlog.debug("copy: serial execution\n");
    rng::copy(in, out);
    fence();
  }
}

/// Copy
template <lib::distributed_iterator DI_IN>
void copy(DI_IN &&first, DI_IN &&last, lib::distributed_iterator auto &&out) {
  mhp::copy(rng::subrange(first, last), out);
}

//
//
// for_each
//
//

/// Collective for_each on distributed range
template <typename ExecutionPolicy>
void for_each(ExecutionPolicy &&policy, lib::distributed_range auto &&dr,
              auto op) {
  if constexpr (std::is_same_v<std::remove_cvref_t<ExecutionPolicy>,
                               device_policy>) {
    lib::drlog.debug("for_each: dpl execution\n");
    for (const auto &s : local_segments(dr)) {

      std::for_each(policy.dpl_policy,
                    lib::__detail::direct_iterator(rng::begin(s)),
                    lib::__detail::direct_iterator(rng::end(s)), op);
    }
  } else {
    lib::drlog.debug("for_each: parallel cpu execution\n");
    for (const auto &s : local_segments(dr)) {
      rng::for_each(s, op);
    }
  }
  barrier();
}

template <lib::distributed_range DR> void for_each(DR &&dr, auto op) {
  for_each(std::execution::par_unseq, std::forward<DR>(dr), op);
}

/// Collective for_each on iterator/sentinel for a distributed range
template <typename ExecutionPolicy, lib::distributed_iterator DI>
void for_each(ExecutionPolicy &&policy, DI first, DI last, auto op) {
  mhp::for_each(std::forward<ExecutionPolicy>(policy),
                rng::subrange(first, last), op);
}

template <lib::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mhp::for_each(std::execution::par_unseq, rng::subrange(first, last), op);
}

//
//
// iota
//
//

/// Collective iota on iterator/sentinel for a distributed range
template <lib::distributed_iterator DI>
void iota(DI first, DI last, auto value) {
  if (default_comm().rank() == 0) {
    std::iota(first, last, value);
  }
  fence();
}

/// Collective iota on distributed range
void iota(lib::distributed_contiguous_range auto &&r, auto value) {
  mhp::iota(rng::begin(r), rng::end(r), value);
}

//
//
// transform
//
//

void transform(lib::distributed_range auto &&in,
               lib::distributed_iterator auto out, auto op) {
  if (aligned(in, out)) {
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::transform(in_seg, rng::begin(out_seg), op);
    }
    barrier();
  } else {
    lib::drlog.debug("transform: serial execution\n");
    rng::transform(in, out, op);
    fence();
  }
}

template <lib::distributed_iterator DI_IN>
void transform(DI_IN &&first, DI_IN &&last,
               lib::distributed_iterator auto &&out, auto op) {
  mhp::transform(rng::subrange(first, last), out, op);
}

} // namespace mhp
