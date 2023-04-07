// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <dr/details/ranges_shim.hpp>
#include <dr/mhp/alignment.hpp>

namespace mhp {

template <rng::viewable_range... R>
class zip_view : public rng::zip_view<R...> {
public:
  template <rng::viewable_range... V>
  zip_view(V &&...v)
      : rng::zip_view<R...>(rng::views::all(std::forward<V>(v))...) {}
};

template <rng::viewable_range... R>
zip_view(R &&...r) -> zip_view<rng::views::all_t<R>...>;

namespace views {

template <rng::viewable_range... R> auto zip(R &&...r) {
  return zip_view(std::forward<R>(r)...);
}

} // namespace views

} // namespace mhp

namespace DR_RANGES_NAMESPACE {} // namespace DR_RANGES_NAMESPACE
