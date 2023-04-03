// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace mhp {

template <typename T, typename Allocator = std::allocator<T>>
class distributed_dense_matrix;

template <typename DM> class dm_rows_iterator;
template <typename DM> class dm_segment;
template <typename DM> class dm_segment_iterator;
template <typename T> class dm_row;

/* template <typename DM> class dm_row_reference {
  using iterator = dm_rows_iterator<DM>;

public:
  using value_type = dm_row<typename DM::value_type>;

  dm_row_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const dm_row_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return *iterator_; }

  dm_segment<DM> &segment() { return (*iterator_).segment(); }

private:
  const iterator iterator_;
}; // dm_row_reference */

template <typename DM>
// class dm_rows_iterator : public lib::normal_distributed_iterator<dm_rows<DM>>
// {
class dm_rows_iterator {
public:
  using value_type = typename mhp::dm_row<typename DM::value_type>;
  using size_type = typename mhp::dm_row<typename DM::value_type>;
  using difference_type = typename DM::difference_type;

  dm_rows_iterator() = default;
  dm_rows_iterator(DM *dm, std::size_t segment_index,
                   std::size_t index) noexcept {
    dm_ = dm;
    rank_ = segment_index;
    index_ = index;
  }

  // dereference
  value_type &operator*() { return dm_->dm_rows_[index_]; }
  value_type operator[](difference_type n) { return dm_->dm_rows_[index_ + n]; }

  // value_type get() const {
  //   auto segment_offset = index_ + dm_->halo_bounds_.prev;
  //   auto value = dm_->win_.template get<value_type>(rank_, segment_offset);
  //   lib::drlog.debug("get {} =  ({}:{})\n", value, rank_, segment_offset);
  //   return value;
  // }

  // void put(const value_type &value) const {
  //   auto segment_offset = index_ + dm_->halo_bounds_.prev;
  //   lib::drlog.debug("put ({}:{}) = {}\n", rank_, segment_offset, value);
  //   dm_->win_.put(value, rank_, segment_offset);
  // }

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

  auto rank() const { return rank_; }
  // auto local() const { return dm_->data_ + index_ + dm_->halo_bounds_.prev; }
  auto segments() const {
    return lib::internal::drop_segments(dm_->segments(), index_);
  }
  auto &halo() const { return dm_->halo(); }

private:
  DM *dm_ = nullptr;
  std::size_t rank_;
  std::size_t index_ = 0;
}; // dm_rows_iterator

template <typename DM> class dm_segment_reference {
  using iterator = dm_segment_iterator<DM>;

public:
  using value_type = typename DM::value_type;

  dm_segment_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const dm_segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // dm_segment_reference

template <typename DM> class dm_segment_iterator {
public:
  using value_type = typename DM::value_type;
  using size_type = typename DM::size_type;
  using difference_type = typename DM::difference_type;

  dm_segment_iterator() = default;
  dm_segment_iterator(DM *dm, std::size_t segment_index, std::size_t index) {
    dm_ = dm;
    rank_ = segment_index;
    index_ = index;
  }

  // Comparison
  bool operator==(const dm_segment_iterator &other) const noexcept {
    return index_ == other.index_ && dm_ == other.dm_;
  }
  auto operator<=>(const dm_segment_iterator &other) const noexcept {
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
  difference_type operator-(const dm_segment_iterator &other) const noexcept {
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
  friend auto operator+(difference_type n, const dm_segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const { return dm_segment_reference<DM>{*this}; }
  auto operator[](difference_type n) const { return *(*this + n); }

  value_type get() const {
    auto segment_offset = index_ + dm_->halo_bounds_.prev;
    auto value = dm_->win_.template get<value_type>(rank_, segment_offset);
    lib::drlog.debug("get {} =  ({}:{})\n", value, rank_, segment_offset);
    return value;
  }

  void put(const value_type &value) const {
    auto segment_offset = index_ + dm_->halo_bounds_.prev;
    lib::drlog.debug("put ({}:{}) = {}\n", rank_, segment_offset, value);
    dm_->win_.put(value, rank_, segment_offset);
  }

  auto rank() const { return rank_; }
  auto local() const { return dm_->data() + index_ + dm_->halo_bounds().prev; }
  auto segments() const {
    return lib::internal::drop_segments(dm_->segments(), index_);
  }
  auto &halo() const { return dm_->halo(); }

private:
  DM *dm_ = nullptr;
  std::size_t rank_;
  std::size_t index_;
}; // dm_segment_iterator

template <typename T> class dm_row_iterator : public std::span<T>::iterator {
public:
  dm_row_iterator(distributed_dense_matrix<T> *dm)
      : std::span<T>::iterator(), dm_(dm){};

  auto segments() const { return dm_->segments(); }
  auto &halo() const { return dm_->halo(); }

private:
  mhp::distributed_dense_matrix<T> *dm_;
};

} // namespace mhp
