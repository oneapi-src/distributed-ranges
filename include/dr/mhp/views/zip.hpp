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

template <typename T>
concept has_rank = requires(T &t) { lib::ranges::rank(t); };

template <typename T>
concept tuple_has_rank = []<std::size_t... N>(std::index_sequence<N...>) {
  return (has_rank<typename std::tuple_element<N, T>::type> || ...);
}(std::make_index_sequence<std::tuple_size_v<T>>());

template <rng::viewable_range... R> class zip_view;

namespace views {

template <rng::viewable_range... R> auto zip(R &&...r) {
  return zip_view(std::forward<R>(r)...);
}

} // namespace views

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
  zip_view(V &&...v)
      : rng_zip_(rng::views::all(v)...), base_(rng::views::all(v)...) {}
  auto begin() const { return zip_iterator(rng::begin(rng_zip_)); }
  auto end() const { return zip_iterator(rng::end(rng_zip_)); }

  //
  // Support for distributed ranges
  //
  auto segments() const {
    // requires at least one distributed range in base
    auto zip_segments = [this](auto &&...base) {
      auto zip_segment = [](auto &&v) {
        auto zip_ranges = [](auto &&...refs) { return views::zip(refs...); };
        return std::apply(zip_ranges, v);
      };

      auto z = rng::views::zip(this->base_segments(base)...) |
               rng::views::transform(zip_segment);
      if (aligned(base...)) {
        return z;
      } else {
        return decltype(z){};
      }
    };

    return std::apply(zip_segments, base_);
  }

  auto rank() const
    requires(tuple_has_rank<std::tuple<R...>>)
  {
    auto select = [](auto &&...v) { return select_rank(v...); };
    return std::apply(select, base_);
  }

private:
  static auto select_rank(auto &&v, auto &&...rest) {
    if constexpr (has_rank<decltype(v)>) {
      return lib::ranges::rank(v);
    } else {
      return select_rank(rest...);
    }
  }

  template <rng::range V> static auto base_segments(V &&base) {
    return lib::ranges::segments(base);
  }

  rng_zip rng_zip_;
  std::tuple<R...> base_;
};

template <rng::viewable_range... R>
zip_view(R &&...r) -> zip_view<rng::views::all_t<R>...>;

} // namespace mhp

namespace DR_RANGES_NAMESPACE {} // namespace DR_RANGES_NAMESPACE

// Needed to satisfy rng::viewable_range
template <rng::random_access_range... V>
inline constexpr bool rng::enable_borrowed_range<mhp::zip_view<V...>> = true;
