// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp {

template <typename T, typename Allocator = std::allocator<T>>
class distributed_dense_matrix;

template <typename T, typename Allocator = std::allocator<T>> class dm_row;

template <typename DM> class dm_rows_iterator {
public:
  using value_type = typename dr::mhp::dm_row<typename DM::value_type>;
  using size_type = typename dr::mhp::dm_row<typename DM::value_type>;
  using difference_type = typename DM::difference_type;

  dm_rows_iterator() = default;
  dm_rows_iterator(DM *dm, std::size_t index) noexcept {
    dm_ = dm;
    index_ = index;
  }

  // dereference
  value_type &operator*() const { return dm_->dm_rows_[index_]; }
  value_type *operator->() const { return &(dm_->dm_rows_[index_]); }

  value_type operator[](difference_type n) {
    difference_type abs_ind = index_ + n;

    if (abs_ind >= dm_->local_rows_indices_.first &&
        abs_ind <= dm_->local_rows_indices_.second) { // regular rows
      return dm_->dm_rows_[index_ + n];
    }
    if (abs_ind >= (difference_type)(dm_->local_rows_indices_.first -
                                     dm_->halo_bounds().prev) &&
        abs_ind < dm_->local_rows_indices_.first) { // halo prev
      return dm_->dm_halo_p_rows_[dm_->halo_bounds().prev -
                                  dm_->local_rows_indices_.first + abs_ind];
    }
    if (abs_ind > dm_->local_rows_indices_.second &&
        abs_ind <= (difference_type)(dm_->local_rows_indices_.second +
                                     dm_->halo_bounds().next)) { // halo next
      return dm_->dm_halo_n_rows_[dm_->halo_bounds().next +
                                  dm_->local_rows_indices_.second - abs_ind];
    }
    assert(0);
  }

  value_type get() const {
    auto segment_offset = index_ + dm_->halo_bounds_.prev;
    auto value = dm_->win_.template get<value_type>(index_, segment_offset);
    dr::drlog.debug("get {} =  ({}:{})\n", value, index_, segment_offset);
    return value;
  }

  void put(const value_type &value) const {
    auto segment_offset = index_ + dm_->halo_bounds_.prev;
    dr::drlog.debug("put ({}:{}) = {}\n", index_, segment_offset, value);
    dm_->win_.put(value, index_, segment_offset);
  }

  // Comparison
  bool operator==(const dm_rows_iterator &other) const noexcept {
    return index_ == other.index_ && dm_ == other.dm_;
  }
  auto operator<=>(const dm_rows_iterator &other) const noexcept {
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
  difference_type operator-(const dm_rows_iterator &other) const noexcept {
    return index_ - other.index_;
  }

  // prefix
  auto &operator++() {
    index_ += 1;
    return *this;
  }
  auto &operator--() {
    index_ -= 1;
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
  friend auto operator+(difference_type n, const dm_rows_iterator &other) {
    return other + n;
  }

  auto segments() const {
    return dr::__detail::drop_segments(dm_->segments(), index_);
  }
  auto &halo() const { return dm_->halo(); }

  bool is_local() { return dm_->is_local_row(index_); }

private:
  DM *dm_ = nullptr;
  std::size_t index_ = 0;
}; // dm_rows_iterator

template <typename T, typename Allocator> class dm_row : public std::span<T> {
  using DM = distributed_dense_matrix<T>;

public:
  using iterator = typename std::span<T>::iterator;

  dm_row(){};
  dm_row(signed long idx, T *ptr, std::size_t size, d_segment<DM> *segment,
         Allocator allocator = Allocator())
      : std::span<T>({ptr, size}), index_(idx), data_(ptr), size_(size),
        segment_(segment){};

  // own memory necessary - the row ist standalone, not part of matrix - index
  // INT_MIN indicates it
  dm_row(std::size_t size)
      : dm_row(INT_MIN, Allocator().allocate(size), size, nullptr) {
    for (std::size_t _i = 0; _i < size_; _i++) {
      data_[_i] = 0;
    }
  }

  // copying ctor
  dm_row(const dm_row &other)
      : std::span<T>({other.data_, other.size_}), index_(other.index_),
        data_(other.data_), size_(other.size_), segment_(other.segment_) {
    assert(other.index_ != INT_MIN);
  }

  // moving ctor
  dm_row(dm_row &&other)
      : std::span<T>({Allocator().allocate(other.size_), other.size_}),
        index_(other.index_), data_(&(*this->begin())), size_(other.size_),
        segment_(other.segment_) {
    assert(other.index_ == INT_MIN);
    // index_ = other.index_;
    // data_ = &(*this->begin());
    // size_ = other.size_;
    // segment_ = other.segment_;

    iterator i = rng::begin(*this), oi = rng::begin(other);
    while (i != this->end()) {
      *(i++) = *(oi++);
    }
  }

  ~dm_row() {
    if (INT_MIN == index_ && nullptr != data_) {
      Allocator().deallocate(data_, size_);
      data_ = nullptr;
      size_ = 0;
      index_ = 0;
    }
  }

  d_segment<DM> *segment() { return segment_; }
  signed long idx() { return index_; }

  T &operator[](int index) { return *(std::span<T>::begin() + index); }

  dm_row<T> operator=(dm_row<T> other) {
    assert(this->size_ == other.size_);
    iterator i = rng::begin(*this), oi = rng::begin(other);
    while (i != this->end()) {
      *(i++) = *(oi++);
    }
    return *this;
  }

private:
  signed long index_ = 0;
  T *data_ = nullptr;
  std::size_t size_ = 0;
  d_segment<DM> *segment_ = nullptr;
};

template <typename DM>
class dm_rows : public std::vector<dm_row<typename DM::value_type>> {
public:
  using iterator = dm_rows_iterator<DM>;
  using value_type = dm_row<typename DM::value_type>;

  dm_rows(DM *dm) { dm_ = dm; }

  auto segments() { return dm_->segments(); }
  auto &halo() { return dm_->halo(); }

  iterator begin() const {
    assert(dm_ != nullptr);
    return dm_rows_iterator(dm_, 0);
  }
  iterator end() const {
    assert(dm_ != nullptr);
    return dm_rows_iterator(dm_, this->size());
  }
  DM *dm() { return dm_; }

private:
  DM *dm_ = nullptr;
};

} // namespace dr::mhp
