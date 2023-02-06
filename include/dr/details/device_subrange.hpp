// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <iterator>
#include <dr/concepts/concepts.hpp>
#include <dr/details/ranges_shim.hpp>

namespace lib {

template <std::forward_iterator I>
class device_subrange : public rng::subrange<I, I>
{
  using base = rng::subrange<I, I>;
public:

  device_subrange() requires std::default_initializable<I> = default;

  constexpr device_subrange(I first, I last, std::size_t rank)
    : base(first, last), rank_(rank) {}

  template <rng::forward_range R>
  constexpr device_subrange(R&& r, std::size_t rank)
    : base(rng::begin(r), rng::end(r)), rank_(rank) {}

  template <lib::remote_range R>
  constexpr device_subrange(R&& r)
    : base(rng::begin(r), rng::end(r)), rank_(lib::ranges::rank(r)) {}

  constexpr std::size_t rank() const noexcept {
    return rank_;
  }

private:
  std::size_t rank_;
};

template <rng::forward_range R>
device_subrange(R&&, std::size_t) -> device_subrange<rng::iterator_t<R>>;

template <lib::remote_range R>
device_subrange(R&&) -> device_subrange<rng::iterator_t<R>>;

} // end lib