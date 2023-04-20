// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "index.hpp"

namespace dr::mhp {

template <typename DM> class segment;
template <typename DM> class segment_iterator;

using key_type = index<>;

template <typename DM> class segment_reference {
  using iterator = segment_iterator<DM>;

public:
  using value_type = typename DM::value_type;

  segment_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // segment_reference

template <typename DM> class segment_iterator {
public:
  using value_type = typename DM::value_type;
  using size_type = typename DM::size_type;
  using difference_type = typename DM::difference_type;

  segment_iterator() = default;
  segment_iterator(DM *dm, std::size_t segment_index, std::size_t index) {
    dm_ = dm;
    rank_ = segment_index;
    index_ = index;
  }

  // Comparison
  bool operator==(const segment_iterator &other) const noexcept {
    return index_ == other.index_ && dm_ == other.dm_;
  }
  auto operator<=>(const segment_iterator &other) const noexcept {
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
  difference_type operator-(const segment_iterator &other) const noexcept {
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
  friend auto operator+(difference_type n, const segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const { return segment_reference<DM>{*this}; }
  auto operator[](difference_type n) const { return *(*this + n); }

  value_type get() const {
    auto segment_offset = index_ + dm_->halo_bounds_.prev;
    auto value = dm_->win_.template get<value_type>(rank_, segment_offset);
    // dr::drlog.debug("get {} =  ({}:{})\n", value, rank_, segment_offset);
    return value;
  }

  void put(const value_type &value) const {
    auto segment_offset = index_ + dm_->halo_bounds_.prev;
    // dr::drlog.debug("put ({}:{}) = {}\n", rank_, segment_offset, value);
    dm_->win_.put(value, rank_, segment_offset);
  }

  auto rank() const { return rank_; }
  auto local() const { return dm_->data_ + index_ + dm_->halo_bounds_.prev; }
  auto segments() const {
    return dr::__detail::drop_segments(dm_->segments(), index_);
  }
  auto &halo() const { return dm_->halo(); }

private:
  DM *dm_ = nullptr;
  std::size_t rank_;
  std::size_t index_;
}; // class dm_segment_iterator

template <typename DM> class segment {
private:
  using iterator = segment_iterator<DM>;
  using value_type = typename DM::value_type;

public:
  using difference_type = std::ptrdiff_t;
  segment() = default;
  segment(DM *dm, std::size_t segment_index, std::size_t size /*, value_type *ptr, key_type shape, */ ) {
    dm_ = dm;
    // ptr_ = ptr;
    // shape_ = shape;
    rank_ = segment_index;
    size_ = size; // shape[0] * shape[1];
  }

  auto size() const { return size_; }

  auto begin() const { return iterator(dm_, rank_, 0); }
  auto end() const { return begin() + size(); }

  auto operator[](difference_type n) const { return *(begin() + n); }

  bool is_local() { return rank_ == default_comm().rank(); }

  key_type shape() { 
    return key_type ({size_ / dm_->shape()[1], dm_->shape()[1]});  
  }

  DM *dm() { return dm_; }

private:
  DM *dm_;
//   value_type *ptr_;
//   key_type shape_;
  std::size_t rank_;
  std::size_t size_;
}; // class segment

} // namespace dr::mhp