// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace shp {

namespace detail {

// Factor n into 2 roughly equal factors
// n = pq, p >= q
std::tuple<std::size_t, std::size_t> factor(size_t n) {
  size_t q = std::sqrt(n);

  while (q > 1 && n / q != static_cast<double>(n) / q) {
    q--;
  }
  size_t p = n / q;

  return {p, q};
}

} // namespace detail

} // namespace shp
