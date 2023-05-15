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
  assert(aligned(in, out));

  auto zip = mhp::views::zip(in, rng::subrange(out, out + rng::size(in)));
  auto transform_op = [op](auto pair) {
    auto &[in, out] = pair;
    out = op(in);
  };
  for_each(zip, transform_op);
}

template <dr::distributed_iterator DI_IN>
void transform(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out,
               auto op) {
  mhp::transform(rng::subrange(first, last), out, op);
}

} // namespace dr::mhp
