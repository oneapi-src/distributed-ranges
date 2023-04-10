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

template <typename T>
concept tuple_has_distributed_range =
    []<std::size_t... N>(std::index_sequence<N...>) {
      return (lib::distributed_range<typename std::tuple_element<N, T>::type> ||
              ...);
    }(std::make_index_sequence<std::tuple_size_v<T>>());

template <typename T>
concept tuple_has_distributed_iterator = []<std::size_t... N>(
                                             std::index_sequence<N...>) {
  return (lib::distributed_iterator<typename std::tuple_element<N, T>::type> ||
          ...);
}(std::make_index_sequence<std::tuple_size_v<T>>());

template <rng::viewable_range... R> class zip_view;

namespace views {

template <rng::viewable_range... R> auto zip(R &&...r) {
  return zip_view(std::forward<R>(r)...);
}

} // namespace views

template <rng::viewable_range... R> class zip_view {
private:
  using rng_zip = rng::zip_view<R...>;
  using rng_zip_iterator = rng::iterator_t<rng_zip>;
  using base_type = std::tuple<R...>;
  using iterator_base_type = std::tuple<rng::iterator_t<R>...>;

public:
  // Wrap the iterator for rng::zip
  class zip_iterator {
  public:
    using value_type = std::iter_value_t<rng_zip_iterator>;
    using difference_type = std::iter_difference_t<rng_zip_iterator>;

    zip_iterator() {}
    zip_iterator(const zip_view *parent, difference_type offset)
        : parent_(parent), offset_(offset) {}

    auto operator+(difference_type n) {
      return zip_iterator(parent_, offset_ + n);
    }
    friend auto operator+(difference_type n, const zip_iterator &other) {
      return other + n;
    }
    auto operator-(difference_type n) {
      return zip_iterator(parent_, offset_ - n);
    }

    auto &operator+=(difference_type n) {
      offset_ += n;
      return *this;
    }
    auto &operator-=(difference_type n) {
      offset_ -= n;
      return *this;
    }
    auto &operator++() {
      offset_++;
      return *this;
    }
    auto operator++(int) {
      auto old = *this;
      offset_++;
      return old;
    }
    auto &operator--() {
      offset_--;
      return *this;
    }
    auto operator--(int) {
      auto old = *this;
      offset_--;
      return old;
    }

    auto operator==(zip_iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ == other.offset_;
    }
    auto operator<=>(zip_iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ <=> other.offset_;
    }

    // Underlying iterator does not return a reference
    auto operator*() const {
      return *(rng::begin(parent_->rng_zip_) + offset_);
    }
    auto operator[](difference_type n) { return *(*this + n); }

    //
    // Support for distributed ranges
    //
    // distributed iterator provides segments
    // remote iterator provides local
    //
    auto segments()
      requires(tuple_has_distributed_range<base_type>)
    {
      return lib::internal::drop_segments(parent_->segments(), offset_);
    }
    auto local() {
      auto remainder = rng::distance(*parent_) - offset_;
      auto offset = offset_;
      auto localize = [remainder, offset](auto... base_iters) {
        // for each base iter, construct a range that covers the rest
        // of the segment, and then zip the ranges together. Return
        // the begin iterator of the zip.
        //
        // The types of this zip_view is different from the containing
        // zip_view, so use rng::zip_view to force template argument
        // type deduction.
        return rng::begin(rng::zip_view(rng::views::counted(
            lib::ranges::local(base_iters) + offset, remainder)...));
      };
      return std::apply(localize, rng::begin(*parent_).base());
    }

  private:
    // return a tuple of iterators for the components
    auto base() {
      auto advance = [this](auto &&...base_ranges) {
        return std::tuple((rng::begin(base_ranges) + this->offset_)...);
      };
      return std::apply(advance, parent_->base_);
    }

    const zip_view *parent_;
    difference_type offset_;
  };

  template <rng::viewable_range... V>
  zip_view(V &&...v)
      : rng_zip_(rng::views::all(v)...), base_(rng::views::all(v)...) {}
  auto begin() const { return zip_iterator(this, 0); }
  auto end() const { return zip_iterator(this, rng::distance(rng_zip_)); }

  //
  // Support for distributed ranges
  //
  // distributed range provides segments
  // remote range provides rank
  //
  auto segments() const
    requires(tuple_has_distributed_range<base_type>)
  {
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
    requires(tuple_has_rank<base_type>)
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
  base_type base_;
};

template <rng::viewable_range... R>
zip_view(R &&...r) -> zip_view<rng::views::all_t<R>...>;

} // namespace mhp

namespace DR_RANGES_NAMESPACE {} // namespace DR_RANGES_NAMESPACE

// Needed to satisfy rng::viewable_range
template <rng::random_access_range... V>
inline constexpr bool rng::enable_borrowed_range<mhp::zip_view<V...>> = true;
