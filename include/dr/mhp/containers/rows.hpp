// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp {

using key_type = index<>;

template <typename T, typename Allocator = std::allocator<T>>
class distributed_dense_matrix;

template <typename DM> class dv_segment;

template <typename T> class dm_row : public std::span<T> {
  using dmatrix = distributed_dense_matrix<T>;
  using dsegment = dv_segment<dmatrix>;

public:
  using iterator = typename std::span<T>::iterator;

  dm_row(){};
  dm_row(signed long idx, T *ptr, std::size_t size, dsegment *segment)
      : std::span<T>({ptr, size}), index_(idx), segment_(segment){};

  dsegment *segment() { return segment_; }
  signed long idx() { return index_; }

  // T &operator[](int index) { return *(std::span<T>::begin() + index); }

  dm_row<T> operator=(dm_row<T> other) {
    assert(rng::distance(*this) == rng::distance(other));
    iterator i = rng::begin(*this), oi = rng::begin(other);
    while (i != this->end()) {
      *(i++) = *(oi++);
    }
    return *this;
  }

private:
  signed long index_ = INT_MIN;
  dsegment *segment_ = nullptr;
};

template <typename DM> class dm_rows_iterator {
public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = typename dr::mhp::dm_row<typename DM::value_type>;
  // using size_type = typename dr::mhp::dm_row<typename DM::value_type>;
  using difference_type = typename DM::difference_type;

  dm_rows_iterator() = default;
  dm_rows_iterator(DM *dm, std::size_t index) noexcept {
    dm_ = dm;
    row_idx_ = index;
  }

  auto operator*() const { return dm_->dm_rows_[row_idx_]; }
  auto operator[](difference_type n) {
    difference_type abs_ind = row_idx_ + n;

    if (abs_ind >= dm_->local_rows_indices_.first &&
        abs_ind <= dm_->local_rows_indices_.second) { // regular rows
      return dm_->dm_rows_[row_idx_ + n];
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

  // value_type get() const {
  //   auto segment_offset = index_ + dm_->halo_bounds_.prev;
  //   auto value = dm_->win_.template get<value_type>(rank_, segment_offset);
  //   dr::drlog.debug("get {} =  ({}:{})\n", value, rank_, segment_offset);
  //   return value;
  // }

  // void put(const value_type &value) const {
  //   auto segment_offset = index_ + dm_->halo_bounds_.prev;
  //   dr::drlog.debug("put ({}:{}) = {}\n", rank_, segment_offset, value);
  //   dm_->win_.put(value, rank_, segment_offset);
  // }

  // Comparison
  bool operator==(const dm_rows_iterator &other) const noexcept {
    return row_idx_ == other.row_idx_ && dm_ == other.dm_;
  }
  auto operator<=>(const dm_rows_iterator &other) const noexcept {
    return row_idx_ <=> other.row_idx_;
  }

  // Only these arithmetics manipulate internal state
  auto &operator-=(difference_type n) {
    row_idx_ -= n;
    return *this;
  }
  auto &operator+=(difference_type n) {
    row_idx_ += n;
    return *this;
  }
  difference_type operator-(const dm_rows_iterator &other) const noexcept {
    return row_idx_ - other.row_idx_;
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
  friend auto operator+(difference_type n, const dm_rows_iterator &other) {
    return other + n;
  }

  auto segments() const {
    // return dr::__details__::drop_segments(dm_->segments(), row_idx_);
    return dm_->segments();
  }
  auto &halo() const { return dm_->halo(); }

  bool is_local() { return dm_->is_local_row(row_idx_); }

  DM *dm() { return dm_; }

private:
  DM *dm_ = nullptr;
  std::size_t row_idx_ = 0;
}; // dm_rows_iterator

template <typename DM>
class dm_rows : public std::vector<dm_row<typename DM::value_type>> {
public:
  using iterator = dm_rows_iterator<DM>;
  using value_type = dm_row<typename DM::value_type>;

  dm_rows(DM *dm) { dm_ = dm; }

  iterator begin() const { return iterator(dm_, 0); }
  iterator end() const { return iterator(dm_, this->size()); }

  auto segments() { return dm_->segments(); }
  auto &halo() { return dm_->halo(); }

  // DM *dm() { return dm_; }

private:
  DM *dm_ = nullptr;
};

} // namespace dr::mhp
