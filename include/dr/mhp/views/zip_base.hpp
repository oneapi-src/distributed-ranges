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
    iter.delta_ += n;
    return iter;
  }
  friend auto operator+(difference_type n, const zip_base_iterator &other) {
    return other + n;
  }
  auto operator-(difference_type n) const {
    auto iter(*this);
    iter.rng_iter_ -= n;
    iter.delta_ -= n;
    return iter;
  }
  auto operator-(zip_base_iterator other) const {
    return rng_iter_ - other.rng_iter_;
  }

  auto &operator+=(difference_type n) {
    rng_iter_ += n;
    delta_ += n;
    return *this;
  }
  auto &operator-=(difference_type n) {
    rng_iter_ -= n;
    delta_ -= n;
    return *this;
  }
  auto &operator++() {
    rng_iter_++;
    delta_++;
    return *this;
  }
  auto operator++(int) {
    auto iter(*this);
    rng_iter_++;
    delta_++;
    return iter;
  }
  auto &operator--() {
    rng_iter_--;
    delta_--;
    return *this;
  }
  auto operator--(int) {
    auto iter(*this);
    rng_iter_--;
    delta_--;
    return iter;
  }

  auto operator==(zip_base_iterator other) const {
    return rng_iter_ == other.rng_iter_;
  }
  auto operator<=>(zip_base_iterator other) const {
    return delta_ <=> other.delta_;
  }

  // Underlying zip_base_iterator does not return a reference
  auto operator*() const { return *rng_iter_; }
  auto operator[](difference_type n) const { return rng_iter_[n]; }

private:
  RngIter rng_iter_;
  std::tuple<BaseIters...> base_;
  difference_type delta_ = 0;
};

template <rng::viewable_range... R> class zip_base_view {
private:
  using rng_zip = rng::zip_view<R...>;
  using rng_zip_iterator = rng::iterator_t<rng_zip>;
  using difference_type = std::iter_difference_t<rng_zip_iterator>;

public:
  template <rng::viewable_range... V>
  zip_base_view(V &&...v)
      : rng_zip_(rng::views::all(v)...), base_(rng::views::all(v)...) {}

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

  auto base() { return base_; }

private:
  rng_zip rng_zip_;
  std::tuple<rng::views::all_t<R>...> base_;
};

template <rng::viewable_range... R>
zip_base_view(R &&...r) -> zip_base_view<rng::views::all_t<R>...>;

} // namespace dr::mhp

namespace dr::mhp::views {

template <rng::viewable_range... R> auto zip_base(R &&...r) {
  return zip_base_view(std::forward<R>(r)...);
}

} // namespace dr::mhp::views
