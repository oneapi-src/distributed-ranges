// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/detail/ranges_shim.hpp>
#include <dr/views/iota.hpp>
#include <dr/views/transform.hpp>

namespace DR_RANGES_NAMESPACE {

template <typename F, typename S> auto local_(rng::common_pair<F, S> p) {
  return rng::make_common_pair(dr::ranges::local(p.first),
                               dr::ranges::local(p.second));
}

} // namespace DR_RANGES_NAMESPACE

namespace dr::mhp {

// Select segments local to this rank and convert the iterators in the
// segment to local
template <typename R> auto local_segments(R &&dr) {
  auto is_local = [](const auto &segment) {
    dr::drlog.debug("is local? its:{} our:{}\n", dr::ranges::rank(segment),
                    default_comm().rank());
    return dr::ranges::rank(segment) == default_comm().rank();
  };

  // Convert from remote iter to local iter
  auto local_iter = [](const auto &segment) {
    dr::drlog.debug("wrapuje segment tr(local), size-of-input-segment:{}\n",
                    rng::size(segment));
    auto retVal = segment | rng::views::transform([](const auto &&v) {
                    // type of v const
                    // ranges::subrange<dr::mhp::distributed_vector<int>::iterator,
                    // dr::mhp::distributed_vector<int>::iterator,
                    // ranges::subrange_kind::sized>&&
                    dr::drlog.debug("konwertuje wartosc do local\n");
                    auto x = dr::ranges::local(v);
                    // type of x ranges::subrange<int*, int*,
                    // ranges::subrange_kind::sized>
                    // static_assert(std::same_as<decltype(x), int>);
                    return x;
                  });
    dr::drlog.debug(
        "wrapwanie segmentu tr(local) skonczone, size-of-output-segment:{}\n",
        rng::size(retVal));
    return retVal;
  };
  return dr::ranges::segments(std::forward<R>(dr)) |
         rng::views::filter(is_local) | rng::views::transform(local_iter);
}

} // namespace dr::mhp

namespace dr::mhp::views {

inline constexpr auto all = rng::views::all;
inline constexpr auto drop = rng::views::drop;
inline constexpr auto iota = dr::views::iota;
inline constexpr auto take = rng::views::take;
inline constexpr auto transform = dr::views::transform;

} // namespace dr::mhp::views
