// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/views/iota.hpp>

namespace dr::mhp {

/// Copy
void copy(dr::distributed_range auto &&in, dr::distributed_iterator auto out) {
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

// copy for dr::views::iota
// views::iota require copying elements ony by one
template <std::integral T>
void copy(std::size_t root, rng::iota_view<T, T> &in,
          dr::distributed_contiguous_iterator auto out) {

  if (default_comm().rank() == root) {
    auto s_itr = dr::ranges::segments(out).begin();
    auto l_itr = (*s_itr).begin();

    for (auto i : in) {
      (l_itr++).put(i);
      if (l_itr == (*s_itr).end()) {
        s_itr++;
        l_itr = (*s_itr).begin();
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
