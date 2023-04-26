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

namespace dr::mhp {

template <typename... Rs> class zip_base_view;

namespace views {

template <typename... Rs> auto zip_base(Rs &&...rs) {
  return zip_base_view(std::forward<Rs>(rs)...);
}

} // namespace views

namespace __detail {

template <typename Base> auto base_to_segments(Base &&base) {
  // Given segments, return elementwise zip
  auto zip_segments = [](auto &&...segments) {
    return views::zip_base(segments...);
  };

  // Given a tuple of segments, return a single segment by doing
  // elementwise zip
  auto zip_segment_tuple = [zip_segments](auto &&v) {
    return std::apply(zip_segments, v);
  };

  // Given base ranges, return segments
  auto bases_to_segments = [zip_segment_tuple](auto &&...bases) {
    auto z = rng::views::zip(dr::ranges::segments(bases)...) |
             rng::views::transform(zip_segment_tuple);
    // return empty segment when ranges are not aligned
    if (aligned(bases...)) {
      return z;
    } else {
      return decltype(z){};
    }
  };

  return std::apply(bases_to_segments, base);
}

} // namespace __detail

template <typename RngIter, typename... BaseIters> class zip_base_iterator {
public:
  using value_type = rng::iter_value_t<RngIter>;
  using difference_type = rng::iter_difference_t<RngIter>;

  using iterator_category = std::random_access_iterator_tag;

  zip_base_iterator() {}
  zip_base_iterator(RngIter rng_iter, BaseIters... base_iters)
      : rng_iter_(rng_iter), base_(base_iters...) {}

  auto operator+(difference_type n) const {
    auto iter(*this);
    iter.rng_iter_ += n;
    iter.offset_ += n;
    return iter;
  }
  friend auto operator+(difference_type n, const zip_base_iterator &other) {
    return other + n;
  }
  auto operator-(difference_type n) const {
    auto iter(*this);
    iter.rng_iter_ -= n;
    iter.offset_ -= n;
    return iter;
  }
  auto operator-(zip_base_iterator other) const {
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

  auto operator==(zip_base_iterator other) const {
    return rng_iter_ == other.rng_iter_;
  }
  auto operator<=>(zip_base_iterator other) const {
    return offset_ <=> other.offset_;
  }

  // Underlying zip_base_iterator does not return a reference
  auto operator*() const { return *rng_iter_; }
  auto operator[](difference_type n) const { return rng_iter_[n]; }

  //
  // Distributed Ranges support
  //
  auto segments() const
    requires(distributed_iterator<BaseIters> && ...)
  {
    return dr::__detail::drop_segments(__detail::base_to_segments(base_),
                                       offset_);
  }

  auto local() const
    requires(remote_iterator<BaseIters> && ...)
  {
    // Create a temporary zip_view and return the iterator. This code
    // assumes the iterator is valid even if the underlying zip_view
    // is destroyed.
    auto zip = [this]<typename... Iters>(Iters &&...iters) {
      return rng::begin(rng::views::zip(rng::subrange(
          dr::ranges::local(std::forward<Iters>(iters)) + this->offset_,
          decltype(dr::ranges::local(iters)){})...));
    };

    return std::apply(zip, base_);
  }

private:
  RngIter rng_iter_;
  std::tuple<BaseIters...> base_;
  difference_type offset_ = 0;
};

template <typename... Rs> class zip_base_view : public rng::view_base {
private:
  using rng_zip = rng::zip_view<Rs...>;
  using rng_zip_iterator = rng::iterator_t<rng_zip>;
  using difference_type = std::iter_difference_t<rng_zip_iterator>;

public:
  template <typename... Vs>
  zip_base_view(Vs &&...vs)
      : rng_zip_(rng::views::all(vs)...), base_(rng::views::all(vs)...) {}

  auto begin() const {
    auto make_begin = [this](auto &&...bases) {
      return zip_base_iterator(rng::begin(this->rng_zip_),
                               rng::begin(bases)...);
    };
    return std::apply(make_begin, base_);
  }
  auto end() const {
    auto make_end = [this](auto &&...bases) {
      return zip_base_iterator(rng::end(this->rng_zip_), rng::end(bases)...);
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
    requires(distributed_range<Rs> && ...)
  {
    return __detail::base_to_segments(base_);
  }

  auto rank() const
    requires(remote_range<Rs> && ...)
  {
    return dr::ranges::rank(std::get<0>(base_));
  }

  auto local() const
    requires(remote_range<Rs> && ...)
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
zip_base_view(Rs &&...rs) -> zip_base_view<rng::views::all_t<Rs>...>;

} // namespace dr::mhp
