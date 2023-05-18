
// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp {

using key_type = index<>;

template <typename DV> class d_segment_iterator;

template <typename DV> class d_segment_reference {
  using iterator = d_segment_iterator<DV>;

public:
  using value_type = typename DV::value_type;

  d_segment_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const d_segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // d_segment_reference

template <typename DV> class d_segment_iterator {
public:
  using value_type = typename DV::value_type;
  using size_type = typename DV::size_type;
  using difference_type = typename DV::difference_type;

  d_segment_iterator() = default;
  d_segment_iterator(DV *dv, std::size_t segment_index, std::size_t index) {
    dv_ = dv;
    segment_index_ = segment_index;
    index_ = index;
  }

  // Comparison
  bool operator==(const d_segment_iterator &other) const noexcept {
    return index_ == other.index_ && dv_ == other.dv_;
  }
  auto operator<=>(const d_segment_iterator &other) const noexcept {
    return index_ <=> other.index_;
  }

  // Only these arithmetics manipulate internal state
  auto &operator-=(difference_type n) {
    index_ -= n;
    return *this;
  }
  auto &operator+=(difference_type n) {
    index_ += n;
    return *this;
  }
  difference_type operator-(const d_segment_iterator &other) const noexcept {
    return index_ - other.index_;
  }

  // prefix
  auto &operator++() {
    *this += 1;
    return *this;
  }
  auto &operator--() {
    *this -= 1;
    return *this;
  }

  // postfix
  auto operator++(int) {
    auto prev = *this;
    *this += 1;
    return prev;
  }
  auto operator--(int) {
    auto prev = *this;
    *this -= 1;
    return prev;
  }

  auto operator+(difference_type n) const {
    auto p = *this;
    p += n;
    return p;
  }
  auto operator-(difference_type n) const {
    auto p = *this;
    p -= n;
    return p;
  }

  // When *this is not first in the expression
  friend auto operator+(difference_type n, const d_segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const { return d_segment_reference<DV>{*this}; }
  auto operator[](difference_type n) const { return *(*this + n); }

  void get(value_type *dst, std::size_t size) const {
    auto segment_offset = index_ + dv_->halo_bounds_.prev;
    dv_->win_.get(dst, size * sizeof(*dst), segment_index_,
                  segment_offset * sizeof(*dst));
  }

  value_type get() const {
    value_type val;
    get(&val, 1);
    return val;
  }

  void put(const value_type *dst, std::size_t size) const {
    auto segment_offset = index_ + dv_->halo_bounds_.prev;
    dr::drlog.debug("dv put:: ({}:{}:{})\n", segment_index_, segment_offset,
                    size);
    dv_->win_.put(dst, size * sizeof(*dst), segment_index_,
                  segment_offset * sizeof(*dst));
  }

  void put(const value_type &value) const { put(&value, 1); }

  auto rank() const { return segment_index_; }
  auto local() const { return dv_->data_ + index_ + dv_->halo_bounds_.prev; }
  auto segments() const {
    return dr::__detail::drop_segments(dv_->segments(), segment_index_, index_);
  }
  auto &halo() const { return dv_->halo(); }

private:
  DV *dv_ = nullptr;
  std::size_t segment_index_;
  std::size_t index_;
}; // d_segment_iterator

template <typename DV> class d_segment {
private:
  using iterator = d_segment_iterator<DV>;

public:
  using difference_type = std::ptrdiff_t;
  d_segment() = default;
  d_segment(DV *dv, std::size_t segment_index, std::size_t size) {
    dv_ = dv;
    segment_index_ = segment_index;
    size_ = size;
  }

  auto size() const { return size_; }

  auto begin() const { return iterator(dv_, segment_index_, 0); }
  auto end() const { return begin() + size(); }

  auto operator[](difference_type n) const { return *(begin() + n); }

  bool is_local() { return segment_index_ == default_comm().rank(); }

  key_type shape() {
    return key_type({size_ / dv_->shape()[1], dv_->shape()[1]});
  }

private:
  DV *dv_;
  std::size_t segment_index_;
  std::size_t size_;
}; // d_segment

} // namespace dr::mhp
