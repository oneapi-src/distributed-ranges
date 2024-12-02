// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/ranges_shim.hpp>

namespace dr::__detail {

template <std::random_access_iterator Iter> class multiply_iterator {
public:
  using value_type = std::iter_value_t<Iter>;
  using difference_type = long long;
  using iterator = multiply_iterator<Iter>;
  using reference = value_type;

  using pointer = iterator;

  using iterator_category = std::random_access_iterator_tag;

  multiply_iterator(Iter iter, std::size_t len, long long pos) noexcept
      : iter_(iter), len_(len), pos_(pos) {}
  multiply_iterator() noexcept = default;
  ~multiply_iterator() noexcept = default;
  multiply_iterator(const multiply_iterator &) noexcept = default;
  multiply_iterator &operator=(const multiply_iterator &) noexcept = default;

  bool operator==(const multiply_iterator &other) const noexcept {
    return iter_ == other.iter_ && pos_ == other.pos_ && len_ == other.len_;
  }

  bool operator!=(const multiply_iterator &other) const noexcept {
    return iter_ != other.iter_ || pos_ != other.pos_ || len_ != other.len_;
  }

  iterator operator+(difference_type offset) const noexcept {
    return iterator(iter_, len_, pos_ + offset);
  }

  iterator operator-(difference_type offset) const noexcept {
    return iterator(iter_, len_, pos_ + offset);
  }

  difference_type operator-(iterator other) const noexcept {
    return pos_ - other.pos_;
  }

  bool operator<(iterator other) const noexcept { return pos_ < other.pos_; }

  bool operator>(iterator other) const noexcept { return pos_ > other.pos_; }

  bool operator<=(iterator other) const noexcept { return pos_ <= other.pos_; }

  bool operator>=(iterator other) const noexcept { return pos_ >= other.pos_; }

  iterator &operator++() noexcept {
    ++pos_;
    return *this;
  }

  iterator operator++(int) noexcept {
    iterator other = *this;
    ++(*this);
    return other;
  }

  iterator &operator--() noexcept {
    --pos_;
    return *this;
  }

  iterator operator--(int) noexcept {
    iterator other = *this;
    --(*this);
    return other;
  }

  iterator &operator+=(difference_type offset) noexcept {
    pos_ += offset;
    return *this;
  }

  iterator &operator-=(difference_type offset) noexcept {
    pos_ -= offset;
    return *this;
  }

  reference operator*() const noexcept { return *(iter_ + (pos_ % len_)); }

  reference operator[](difference_type offset) const noexcept {
    return *(*this + offset);
  }

  friend iterator operator+(difference_type n, iterator iter) {
    return iter.pos_ + n;
  }

  auto local() const
    requires(dr::ranges::__detail::has_local<Iter>)
  {
    auto iter = dr::ranges::__detail::local(iter_);
    return multiply_iterator<decltype(iter)>(std::move(iter), len_, pos_);
  }

private:
  Iter iter_;
  std::size_t len_;
  long long pos_;
};

template <rng::random_access_range V>
  requires(rng::sized_range<V>)
class multiply_view : public rng::view_interface<multiply_view<V>> {
public:
  template <rng::viewable_range R>
  multiply_view(R &&r, std::size_t n)
      : base_(rng::views::all(std::forward<R>(r))), n_(n) {}

  auto begin() const {
    return multiply_iterator(rng::begin(base_), base_.size(), 0);
  }

  auto end() const {
    return multiply_iterator(rng::begin(base_), base_.size(),
                             n_ * base_.size());
  }

  auto size() const { return rng::size(base_); }

private:
  V base_;
  std::size_t n_;
};

template <rng::viewable_range R>
multiply_view(R &&r, std::size_t n) -> multiply_view<rng::views::all_t<R>>;

} // namespace dr::__detail
