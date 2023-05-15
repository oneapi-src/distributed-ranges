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
