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

template <rng::viewable_range... R> class zip_view {
  using rng_zip = rng::zip_view<R...>;
  using rng_zip_iterator = decltype(rng::begin(std::declval<rng_zip>()));

public:
  // Wrap the iterator for rng::zip
  template <typename RngIter> class zip_iterator {
  public:
    using value_type = std::iter_value_t<RngIter>;
    using difference_type = std::iter_difference_t<RngIter>;

    zip_iterator() {}
    zip_iterator(RngIter it) : rng_iterator_(it) {}

    auto &operator++() {
      ++rng_iterator_;
      return *this;
    }
    auto operator++(int) {
      auto old = *this;
      rng_iterator_++;
      return old;
    }
    auto &operator--() {
      --rng_iterator_;
      return *this;
    }
    auto operator--(int) {
      auto old = *this;
      rng_iterator_--;
      return old;
    }

    auto operator==(zip_iterator other) const {
      return rng_iterator_ == other.rng_iterator_;
    }
    auto operator<=>(zip_iterator other) const {
      return rng_iterator_ <=> other.rng_iterator_;
    }

    // Underlying iterator does not return a reference
    auto operator*() const { return *rng_iterator_; }

  private:
    RngIter rng_iterator_;
  };

  template <rng::viewable_range... V>
  zip_view(V &&...v) : rng_zip_(rng::views::all(std::forward<V>(v))...) {}
  auto begin() const { return zip_iterator(rng::begin(rng_zip_)); }
  auto end() const { return zip_iterator(rng::end(rng_zip_)); }

private:
  rng_zip rng_zip_;
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

// Needed to satisfy rng::viewable_range
template <rng::random_access_range... V>
inline constexpr bool rng::enable_borrowed_range<mhp::zip_view<V...>> = true;
