// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <concepts>
#include <dr/concepts/concepts.hpp>
#include <dr/mhp/views/zip.hpp>

namespace dr::mhp::_detail {
template <dr::distributed_range R1, dr::distributed_range R2>
  requires std::equality_comparable_with<rng::range_value_t<R1>,
                                         rng::range_value_t<R2>>
bool equal(std::size_t root, bool root_provided, R1 &&r1, R2 &&r2) {
  return true;
}

} // namespace dr::mhp::_detail

namespace dr::mhp {
template <dr::distributed_range R1, dr::distributed_range R2>
  requires std::equality_comparable_with<rng::range_value_t<R1>,
                                         rng::range_value_t<R2>>
bool equal(std::size_t root, R1 &&r1, R2 &&r2) {
  return _detail::equal(root, true, r1, r2);
}

template <dr::distributed_range R1, dr::distributed_range R2>
  requires std::equality_comparable_with<rng::range_value_t<R1>,
                                         rng::range_value_t<R2>>
bool equal(R1 &&r1, R2 &&r2) {
  return _detail::equal(0, false, r1, r2);
}

} // namespace dr::mhp
