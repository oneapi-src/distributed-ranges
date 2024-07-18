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
#include <dr/mp/global.hpp>

namespace dr::mp {

void transform(rng::forward_range auto &&in, dr::distributed_iterator auto out,
               auto op) {
  if (rng::empty(in)) {
    return;
  }
  assert(aligned(in, out));

  auto zip = mp::views::zip(in, rng::subrange(out, out + rng::size(in)));
  auto transform_op = [op](auto pair) {
    auto &[in, out] = pair;
    out = op(in);
  };
  for_each(zip, transform_op);
}

template <rng::forward_iterator DI_IN>
void transform(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out,
               auto op) {
  mp::transform(rng::subrange(first, last), out, op);
}

} // namespace dr::mp
