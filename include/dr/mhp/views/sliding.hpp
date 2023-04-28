// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges.hpp>
#include <dr/detail/segments_tools.hpp>

namespace dr::mhp {
namespace views {
namespace __detail {

struct sliding_fn {
  template <typename Rng>
      requires rng::viewable_range<Rng> && rng::forward_range<Rng>
  auto
  operator()(Rng && rng) const {
    return rng::view::sliding(static_cast<Rng &&>(rng), 2); // fixme: get here halo.prev+halo.next+1 from range
  }
};

}

inline constexpr __detail::sliding_fn sliding{};


}
}