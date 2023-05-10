// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>

namespace dr::mhp {

/// Copy
void copy(dr::distributed_range auto &&in, dr::distributed_iterator auto out) {
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

/// Copy
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

/// Copy
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
