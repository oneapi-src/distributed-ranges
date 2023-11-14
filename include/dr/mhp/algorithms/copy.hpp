// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>

namespace dr::mhp {

/// Copy
void copy(rng::forward_range auto &&in, dr::distributed_iterator auto out) {
  if (rng::empty(in)) {
    return;
  }

  auto copy = [](auto &&v) { std::get<1>(v) = std::get<0>(v); };

  for_each(views::zip(in, views::counted(out, rng::size(in))), copy);
}

/// Copy
template <dr::distributed_iterator DI_IN>
void copy(DI_IN &&first, DI_IN &&last, dr::distributed_iterator auto &&out) {
  copy(rng::subrange(first, last), out);
}

template <std::contiguous_iterator CI_IN>
void copy(CI_IN &&first, CI_IN &&last,
          dr::distributed_contiguous_iterator auto out) {
  copy(0, rng::subrange(first, last), out);
}

/// Copy distributed to local
void copy(std::size_t root, dr::distributed_contiguous_range auto &&in,
          std::contiguous_iterator auto out) {
  if (default_comm().rank() == root) {
    for (const auto &segment : dr::ranges::segments(in)) {
      auto sz = rng::size(segment);
      rng::begin(segment).get(std::to_address(out), sz);
      out += sz;
    }
  }
  barrier();
}

/// Copy local to distributed
void copy(std::size_t root, rng::contiguous_range auto &&in,
          dr::distributed_contiguous_iterator auto out) {
  if (default_comm().rank() == root) {
    auto in_ptr = std::to_address(in.begin());
    for (auto remainder = rng::size(in); remainder > 0;) {
      auto segment = *(dr::ranges::segments(out).begin());
      auto sz = std::min(rng::size(segment), remainder);
      assert(sz > 0);
      rng::begin(segment).put(in_ptr, sz);
      in_ptr += sz;
      out += sz;
      remainder -= sz;
    }
  }
  barrier();
}

} // namespace dr::mhp
