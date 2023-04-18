// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <execution>
#include <type_traits>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

//
//
// fill
//
//

/// Collective fill on distributed range
void fill(dr::distributed_contiguous_range auto &&dr, auto value) {
  for (const auto &s : local_segments(dr)) {
    rng::fill(s, value);
  }
  barrier();
}

/// Collective fill on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
void fill(DI first, DI last, auto value) {
  mhp::fill(rng::subrange(first, last), value);
}

//
//
// copy
//
//

/// Copy
void copy(dr::distributed_contiguous_range auto &&in,
          dr::distributed_iterator auto out) {
  if (aligned(in, out)) {
    dr::drlog.debug("copy: parallel execution\n");
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::copy(in_seg, rng::begin(out_seg));
    }
    barrier();
  } else {
    dr::drlog.debug("copy: serial execution\n");
    rng::copy(in, out);
    fence();
  }
}

/// Copy
template <dr::distributed_iterator DI_IN>
void copy(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out) {
  mhp::copy(rng::subrange(first, last), out);
}

//
//
// for_each
//
//

/// Collective for_each on distributed range
void for_each(dr::distributed_range auto &&dr, auto op) {
#if SYCL_LANGUAGE_VERSION
  if (mhp::use_sycl()) {
    dr::drlog.debug("for_each: dpl execution\n");
    for (const auto &s : local_segments(dr)) {

      std::for_each(dpl_policy(), dr::__detail::direct_iterator(rng::begin(s)),
                    dr::__detail::direct_iterator(rng::end(s)), op);
    }
    barrier();
    return;
  }
#endif

  dr::drlog.debug("for_each: parallel cpu execution\n");
  for (const auto &s : local_segments(dr)) {
    rng::for_each(s, op);
  }
  barrier();
}

/// Collective for_each on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI>
void for_each(DI first, DI last, auto op) {
  mhp::for_each(rng::subrange(first, last), op);
}

//
//
// iota
//
//

/// Collective iota on iterator/sentinel for a distributed range
template <dr::distributed_iterator DI, std::integral T>
void iota(DI first, DI last, T value) {
  if (default_comm().rank() == 0) {
    std::iota(first, last, value);
  }
  fence();
}

/// Collective iota on distributed range
void iota(dr::distributed_range auto &&r, std::integral auto value) {
  mhp::iota(rng::begin(r), rng::end(r), value);
}

//
//
// transform
//
//

void transform(dr::distributed_range auto &&in,
               dr::distributed_iterator auto out, auto op) {
  if (aligned(in, out)) {
    for (const auto &&[in_seg, out_seg] :
         rng::views::zip(local_segments(in), local_segments(out))) {
      rng::transform(in_seg, rng::begin(out_seg), op);
    }
    barrier();
  } else {
    dr::drlog.debug("transform: serial execution\n");
    rng::transform(in, out, op);
    fence();
  }
}

template <dr::distributed_iterator DI_IN>
void transform(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out,
               auto op) {
  mhp::transform(rng::subrange(first, last), out, op);
}

} // namespace dr::mhp
