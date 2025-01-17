// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <concepts>
#include <dr/concepts/concepts.hpp>
#include <dr/mp/algorithms/reduce.hpp>
#include <dr/mp/algorithms/transform.hpp>
#include <dr/mp/views/zip.hpp>

namespace dr::mp::_detail {
template <dr::distributed_range R1, dr::distributed_range R2>
  requires std::equality_comparable_with<rng::range_value_t<R1>,
                                         rng::range_value_t<R2>>
bool equal(std::size_t root, bool root_provided, R1 &&r1, R2 &&r2) {

  if (rng::distance(r1) != rng::distance(r2)) {
    return false;
  }

  // we must use ints instead of bools, because distributed ranges do not
  // support bools
  auto compare = [](auto &&elems) {
    return elems.first == elems.second ? 1 : 0;
  };

  auto zipped_views = views::zip(r1, r2);
  auto compared = dr::mp::views::transform(zipped_views, compare);
  
  auto min = [](double x, double y) { return std::min(x, y); };
  if (root_provided) {
    auto result = mp::reduce(root, compared, 1, min);
    return result == 1;
  }
  auto result = mp::reduce(compared, 1, min);
  return result == 1;
}

} // namespace dr::mp::_detail

namespace dr::mp {
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

} // namespace dr::mp
