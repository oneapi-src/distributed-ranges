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
            base_local(base_iters) + offset, remainder)...));
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

    // If it is not a remote iterator, assume it is a local iterator
    auto static base_local(auto iter) { return iter; }

    auto static base_local(lib::remote_iterator auto iter) {
      return lib::ranges::local(iter);
    }

    const zip_view *parent_;
    difference_type offset_;
  };

  template <rng::viewable_range... V>
  zip_view(V &&...v)
      : rng_zip_(rng::views::all(v)...), base_(rng::views::all(v)...) {
    compute_segment_descriptors(std::forward<V>(v)...);
  }

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
  //
  // Support for distributed ranges
  //
  struct segment_descriptor {
    std::size_t offset, size;
  };

  template <typename... V> void compute_segment_descriptors(V &&...v) {}

  template <typename... V>
  void compute_segment_descriptors(V &&...v)
    requires(lib::distributed_range<V> || ...)
  {
    auto segments = lib::ranges::segments(select_dist_range(v...));
    segment_descriptor descriptor{0, 0};
    for (auto segment : segments) {
      descriptor.size = rng::distance(segment);
      segment_descriptors_.emplace_back(descriptor);
      descriptor.offset += descriptor.size;
    }
  }

  static auto select_rank(auto &&v, auto &&...rest) {
    if constexpr (has_rank<decltype(v)>) {
      return lib::ranges::rank(v);
    } else {
      return select_rank(rest...);
    }
  }

  static auto select_dist_range(auto &&v, auto &&...rest) {
    if constexpr (lib::distributed_range<decltype(v)>) {
      return rng::views::all(v);
    } else {
      return select_dist_range(rest...);
    }
  }

  template <lib::distributed_range V> auto base_segments(V &&base) const {
    return lib::ranges::segments(base);
  }

  // If this is not a distributed range, then assume it is local and
  // segment according to segment_descriptors
  template <rng::range V> auto base_segments(V &&base) const {
    auto make_segment = [base](const segment_descriptor &d) {
      return rng::subrange(rng::begin(base) + d.offset,
                           rng::begin(base) + d.offset + d.size);
    };

    return segment_descriptors_ | rng::views::transform(make_segment);
  }

  // Expensive to copy if there are many segments
  std::vector<segment_descriptor> segment_descriptors_;

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
