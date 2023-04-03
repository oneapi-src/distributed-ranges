// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace mhp {

template <typename DM>
class dm_subrange_iterator
// : public rng::subrange<dm_subrange_iterator<DM>, dm_subrange_iterator<DM>,
//                        rng::subrange_kind::sized>::iterator
{
public:
  using value_type = typename DM::value_type;
  using difference_type = typename DM::difference_type;

  dm_subrange_iterator(DM *dm, std::pair<std::size_t, std::size_t> row_rng,
                       std::pair<std::size_t, std::size_t> col_rng, difference_type index = 0) noexcept {
    dm_ = dm;
    row_rng_ = row_rng;
    col_rng_ = col_rng;
    index_ = index; 
  }
  ~dm_subrange_iterator() noexcept {};

  value_type &operator*() {
    return *(dm_->data() + find_dm_offset(index_));
  }

  value_type operator[](int n) { return *(dm_->data() + find_dm_offset(index_ + n)); }

  value_type & operator[](std::pair<int, int> p) {
      return *(dm_->data() + find_dm_offset(index_) + dm_->shape()[0] * p.second + p.first);
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
    index_ += 1;
    return prev;
  }
  auto operator--(int) {
    auto prev = *this;
    index_ -= 1;
    return prev;
  }

  auto operator+(difference_type n) const {
    return dm_subrange_iterator(dm_, row_rng_, col_rng_, index_ + n);
  }
  auto operator-(difference_type n) const {
    return dm_subrange_iterator(dm_, row_rng_, col_rng_, index_ - n);
  }

  // When *this is not first in the expression
  friend auto operator+(difference_type n, const dm_subrange_iterator &other) {
    return other + n;
  }

  auto &halo() { return dm_->halo(); }
  auto segments() { return dm_->segments(); }

private:
  /*
   * converts index within subrange (viewed as linear contiguous space)
   * into index within physical segment in dm
   */
  std::size_t const find_dm_offset(int index) {
    int ind_rows, ind_cols;
    int offset;

    ind_rows = index / (col_rng_.second - col_rng_.first);
    ind_cols = index % (col_rng_.second - col_rng_.first);

    offset = row_rng_.first * dm_->shape()[0] + col_rng_.first;
    offset += ind_rows * dm_->shape()[0] + ind_cols;

    return offset;
  }

  std::size_t const find_dm_offset_neg(int index) {
    std::cout << "find_dm_offset neg index=" << index ;
    int ind_rows, ind_cols;
    int offset;

    ind_rows = index / (col_rng_.second - col_rng_.first);
    ind_cols = index % (col_rng_.second - col_rng_.first);
    std::cout << " ind_rows = " << ind_rows; 
    std::cout << " ind_cols = " << ind_cols;
    std::cout << " dm_->shape()[0] = " << dm_->shape()[0];

    offset = row_rng_.first * dm_->shape()[0] + col_rng_.first;
    offset += ind_rows * dm_->shape()[0] + ind_cols;

    std::cout << " --> offset=" << offset << std::endl;
    
    return offset;
  }

private:
  DM *dm_;
  std::pair<int, int> row_rng_;
  std::pair<int, int> col_rng_;

  std::size_t index_ = 0;
};

template <typename DM>
class dm_subrange
// : public rng::subrange<dm_subrange_iterator<DM>, dm_subrange_iterator<DM>,
// rng::subrange_kind::sized>
{
public:
  using iterator = dm_subrange_iterator<DM>;
  using value_type = typename DM::value_type;

  dm_subrange(DM &dm, std::pair<std::size_t, std::size_t> row_rng,
              std::pair<std::size_t, std::size_t> col_rng) {
    dm_ = &dm;
    row_rng_ = row_rng;
    col_rng_ = col_rng;

    subrng_size_ =
        (col_rng.second - col_rng.first) * (row_rng.second - row_rng.first);
  }

  iterator begin() const { return iterator(dm_, row_rng_, col_rng_); }
  iterator end() const { return begin() + subrng_size_; }

  auto size() { return subrng_size_; }

  auto &halo() const { return dm_->halo(); }
  auto segments() const { return dm_->segments(); }

private:
  DM *dm_;
  std::pair<std::size_t, std::size_t> row_rng_;
  std::pair<std::size_t, std::size_t> col_rng_;

  std::size_t subrng_size_ = 0;

}; // class subrange

template <typename DM>
void dm_transform(mhp::dm_subrange<DM> &in, mhp::dm_subrange_iterator<DM> out,
                  auto op) {
  for (mhp::dm_subrange_iterator<DM> i = in.begin(); i != in.end(); i++) {
    *(out++) = op(i);
  }
}

} // namespace mhp
