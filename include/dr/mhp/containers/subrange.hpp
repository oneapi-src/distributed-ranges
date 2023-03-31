// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace mhp {

template <typename DM>
class dm_subrange_iterator
    : public rng::subrange<dm_subrange_iterator<DM>, dm_subrange_iterator<DM>,
                           rng::subrange_kind::sized>::iterator {
public:
  using value_type = typename DM::value_type;
  using difference_type = typename DM::difference_type;

  dm_subrange_iterator(DM *dm, std::pair<std::size_t, std::size_t> row_rng,
                       std::pair<std::size_t, std::size_t> col_rng) {
    dm_ = dm;
    row_rng_ = row_rng;
    col_rng_ = col_rng;
    index_ = 0;
  }

  value_type &operator*() {
    return *(dm_->begin() + find_local_offset(index_));
  }

  bool operator==(dm_subrange_iterator &other) {
    return this->index_ == other.index_;
  }
  bool operator!=(dm_subrange_iterator &other) {
    return this->index_ != other.index_;
  }
  auto operator<=>(const dm_subrange_iterator &other) const noexcept {
    return this->index_ <=> other.index_;
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

  difference_type operator-(const dm_subrange_iterator &other) const noexcept {
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
  friend auto operator+(difference_type n, const dm_subrange_iterator &other) {
    return other + n;
  }

private:
  /*
   * converts index within subrange (viewed as linear contiguous space)
   * into index within physical segment in dm
   */
  std::size_t const find_local_offset(std::size_t index) {
    std::size_t ind_rows, ind_cols;
    std::size_t offset;

    ind_rows = index / (col_rng_.second - col_rng_.first);
    ind_cols = index % (col_rng_.second - col_rng_.first);

    offset = row_rng_.first * dm_->shape()[0] + col_rng_.first;
    offset += ind_rows * dm_->shape()[0] + ind_cols;

    return offset / dm_->segsize();
  }

private:
  DM *dm_;
  std::pair<std::size_t, std::size_t> row_rng_;
  std::pair<std::size_t, std::size_t> col_rng_;

  std::size_t index_ = 0;
};

template <typename DM>
class dm_subrange
    : public rng::subrange<dm_subrange_iterator<DM>, dm_subrange_iterator<DM>,
                           rng::subrange_kind::sized> {
public:
  using iterator = dm_subrange_iterator<DM>;
  using value_type = typename DM::value_type;

  // dm_subrange(){};
  // dm_subrange(dm_subrange &){};
  dm_subrange(DM &dm, std::pair<std::size_t, std::size_t> row_rng,
              std::pair<std::size_t, std::size_t> col_rng) {
    dm_ = &dm;
    row_rng_ = row_rng;
    col_rng_ = col_rng;

    subrng_size_ =
        (col_rng.second - col_rng.first) * (row_rng.second - row_rng.first);
  }

  iterator begin() const { return iterator(dm_, row_rng_, col_rng_); }
  iterator end() const {
    return iterator(dm_, row_rng_, col_rng_) + subrng_size_;
  }
  value_type &operator[](int n) { return *(begin() + n); }

  auto size() { return subrng_size_; }

  auto &halo() const { return dm_->halo(); }
  auto segments() const { return dm_->segments(); }

private:
  DM *dm_;
  std::pair<std::size_t, std::size_t> row_rng_;
  std::pair<std::size_t, std::size_t> col_rng_;

  std::size_t subrng_size_ = 0;

}; // class subrange

} // namespace mhp
