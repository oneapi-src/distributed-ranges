// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/alignment.hpp>
#include <dr/mhp/views/segmented.hpp>

namespace dr::mhp::__detail {

template <typename R>
concept zipable = rng::random_access_range<R> && rng::common_range<R>;

} // namespace dr::mhp::__detail

namespace dr::mhp {

template <__detail::zipable... Rs> class zip_view;

namespace views {

template <typename... Rs> auto zip(Rs &&...rs) {
  return zip_view(std::forward<Rs>(rs)...);
}

} // namespace views

namespace __detail {

template <typename T>
concept is_distributed = distributed_range<std::remove_cvref_t<T>> ||
                         distributed_iterator<std::remove_cvref_t<T>>;

template <typename T, typename... Rest>
inline auto select_segments(T &&t, Rest &&...rest) {
  if constexpr (is_distributed<T>) {
    return dr::ranges::segments(std::forward<T>(t));
  } else {
    return select_segments(std::forward<Rest>(rest)...);
  }
}

template <typename T, typename Seg> inline auto tpl_segments(T &&t, Seg &&tpl) {
  if constexpr (is_distributed<T>) {
    return dr::ranges::segments(std::forward<T>(t));
  } else if constexpr (rng::forward_range<T>) {
    return views::segmented(std::forward<T>(t), std::forward<Seg>(tpl));
  } else if constexpr (rng::forward_iterator<T>) {
    return views::segmented(rng::subrange(std::forward<T>(t), T{}),
                            std::forward<Seg>(tpl));
  }
}

template <typename Base> auto base_to_segments(Base &&base) {
  // Given segments, return elementwise zip
  auto zip_segments = [](auto &&...segments) {
    return views::zip(segments...);
  };

  // Given a tuple of segments, return a single segment by doing
  // elementwise zip
  auto zip_segment_tuple = [zip_segments](auto &&v) {
    return std::apply(zip_segments, v);
  };

  // Given base ranges, return segments
  auto bases_to_segments = [zip_segment_tuple](auto &&...bases) {
    bool is_aligned = aligned(bases...);
    auto tpl = select_segments(bases...);
    return rng::views::zip(tpl_segments(bases, tpl)...) |
           rng::views::transform(zip_segment_tuple) |
           rng::views::filter([is_aligned](auto &&v) { return is_aligned; });
  };

  return std::apply(bases_to_segments, base);
}

// based on https://ericniebler.github.io/range-v3/#autotoc_md30  "Create custom
// iterators"
template <typename Iter> struct cursor_over_local_ranges {
  Iter iter;

  auto read() const {
    return rng::subrange(dr::ranges::local(rng::begin(*iter)),
                         dr::ranges::local(rng::begin(*iter)) +
                             rng::size(*iter));
  }

  bool equal(const cursor_over_local_ranges &other) const {
    return iter == other.iter;
  }
  void next() { ++iter; }
  void prev() { --iter; }
  void advance(std::ptrdiff_t n) { this->iter += n; }
  std::ptrdiff_t distance_to(const cursor_over_local_ranges &other) const {
    return other.iter - this->iter;
  }

  cursor_over_local_ranges() = default;
  cursor_over_local_ranges(Iter iter) : iter(iter) {}
};

} // namespace __detail

template <std::random_access_iterator RngIter,
          std::random_access_iterator... BaseIters>
class zip_iterator {
public:
  using value_type = rng::iter_value_t<RngIter>;
  using difference_type = rng::iter_difference_t<RngIter>;

  using iterator_category = std::random_access_iterator_tag;

  zip_iterator() {}
  zip_iterator(RngIter rng_iter, BaseIters... base_iters)
      : rng_iter_(rng_iter), base_(base_iters...) {}

  auto operator+(difference_type n) const {
    auto iter(*this);
    iter.rng_iter_ += n;
    iter.offset_ += n;
    return iter;
  }
  friend auto operator+(difference_type n, const zip_iterator &other) {
    return other + n;
  }
  auto operator-(difference_type n) const {
    auto iter(*this);
    iter.rng_iter_ -= n;
    iter.offset_ -= n;
    return iter;
  }
  auto operator-(zip_iterator other) const {
    return rng_iter_ - other.rng_iter_;
  }

  auto &operator+=(difference_type n) {
    rng_iter_ += n;
    offset_ += n;
    return *this;
  }
  auto &operator-=(difference_type n) {
    rng_iter_ -= n;
    offset_ -= n;
    return *this;
  }
  auto &operator++() {
    rng_iter_++;
    offset_++;
    return *this;
  }
  auto operator++(int) {
    auto iter(*this);
    rng_iter_++;
    offset_++;
    return iter;
  }
  auto &operator--() {
    rng_iter_--;
    offset_--;
    return *this;
  }
  auto operator--(int) {
    auto iter(*this);
    rng_iter_--;
    offset_--;
    return iter;
  }

  auto operator==(zip_iterator other) const {
    return rng_iter_ == other.rng_iter_;
  }
  auto operator<=>(zip_iterator other) const {
    return offset_ <=> other.offset_;
  }

  // Underlying zip_iterator does not return a reference
  auto operator*() const { return *rng_iter_; }
  auto operator[](difference_type n) const { return rng_iter_[n]; }

  //
  // Distributed Ranges support
  //
  auto segments() const
    requires(distributed_iterator<BaseIters> || ...)
  {
    return dr::__detail::drop_segments(__detail::base_to_segments(base_),
                                       offset_);
  }

  auto rank() const
    requires(remote_iterator<BaseIters> || ...)
  {
    return dr::ranges::rank(std::get<0>(base_));
  }

  auto local() const
    requires(remote_iterator<BaseIters> || ...)
  {
    // Create a temporary zip_view and return the iterator. This code
    // assumes the iterator is valid even if the underlying zip_view
    // is destroyed.
    auto zip = [this]<typename... Iters>(Iters &&...iters) {
      return rng::begin(rng::views::zip(
          rng::subrange(base_local(std::forward<Iters>(iters)) + this->offset_,
                        decltype(base_local(iters)){})...));
    };

    return std::apply(zip, base_);
  }

private:
  auto static base_local(auto iter) {
    if constexpr (dr::remote_iterator<decltype(iter)>) {
      return dr::ranges::local(iter);
    } else if constexpr (dr::localizable_contiguous_range<decltype(*iter)>) {
      return rng::basic_iterator<
          dr::mhp::__detail::cursor_over_local_ranges<decltype(iter)>>(iter);
    } else {
      // If it is neither a remote iterator, nor an iterator pointing to range
      // that can made be local, then assume it is a local iterator.
      return iter;
    }
  }

  RngIter rng_iter_;
  std::tuple<BaseIters...> base_;
  difference_type offset_ = 0;
};

template <__detail::zipable... Rs> class zip_view : public rng::view_base {
private:
  using rng_zip = rng::zip_view<Rs...>;
  using rng_zip_iterator = rng::iterator_t<rng_zip>;
  using difference_type = std::iter_difference_t<rng_zip_iterator>;

public:
  zip_view(Rs... rs)
      : rng_zip_(rng::views::all(rs)...), base_(rng::views::all(rs)...) {}

  auto begin() const {
    auto make_begin = [this](auto &&...bases) {
      return zip_iterator(rng::begin(this->rng_zip_), rng::begin(bases)...);
    };
    return std::apply(make_begin, base_);
  }
  auto end() const
    requires(rng::common_range<rng_zip>)
  {
    auto make_end = [this](auto &&...bases) {
      return zip_iterator(rng::end(this->rng_zip_), rng::end(bases)...);
    };
    return std::apply(make_end, base_);
  }
  auto size() const { return rng::size(rng_zip_); }
  auto operator[](difference_type n) const { return rng_zip_[n]; }

  auto base() const { return base_; }

  //
  // Distributed Ranges support
  //
  auto segments() const
    requires(distributed_range<Rs> || ...)
  {
    return __detail::base_to_segments(base_);
  }

  auto rank() const
    requires(remote_range<Rs> || ...)
  {
    return dr::ranges::rank(std::get<0>(base_));
  }

  auto local() const
    requires(remote_range<Rs> || ...)
  {
    auto zip = []<typename... Vs>(Vs &&...bases) {
      return rng::views::zip(dr::ranges::local(std::forward<Vs>(bases))...);
    };

    return std::apply(zip, base_);
  }

private:
  rng_zip rng_zip_;
  std::tuple<rng::views::all_t<Rs>...> base_;
};

template <typename... Rs>
zip_view(Rs &&...rs) -> zip_view<rng::views::all_t<Rs>...>;

} // namespace dr::mhp
