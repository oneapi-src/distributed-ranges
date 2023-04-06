// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace mhp {

template <typename DM> class dm_subrange_iterator {
public:
  using value_type = typename DM::value_type;
  using difference_type = typename DM::difference_type;

  dm_subrange_iterator(){};

  dm_subrange_iterator(DM *dm, std::pair<std::size_t, std::size_t> row_rng,
                       std::pair<std::size_t, std::size_t> col_rng,
                       difference_type index = 0) noexcept {
    dm_ = dm;
    row_rng_ = row_rng;
    col_rng_ = col_rng;
    index_ = index;
  }

  value_type &operator*() const {

    int offset = dm_->halo_bounds().prev * dm_->shape()[0] +
                 find_dm_offset(index_) -
                 default_comm().rank() * dm_->segment_size();

    if ((offset >= (int)dm_->data_size()) || (offset < 0)) {
      fmt::print("{}: index {} offset {} hb {} dm_off {} \n",
                 default_comm().rank(), index_, offset,
                 dm_->halo_bounds().prev * dm_->shape()[1],
                 find_dm_offset(index_));
    }
    assert(offset > 0);
    assert(offset < (int)dm_->data_size());
    return *(dm_->data() + offset);
  }

  value_type &operator[](int n) {
    int offset = dm_->halo_bounds().prev * dm_->shape()[1] +
                 find_dm_offset(index_ + n) -
                 default_comm().rank() * dm_->segment_size();

    assert(offset > 0);
    assert(offset < (int)dm_->data_size());
    return *(dm_->data() + offset);
  }

  value_type &operator[](std::pair<int, int> p) {
    int offset = dm_->halo_bounds().prev * dm_->shape()[1] +
                 find_dm_offset(index_) -
                 default_comm().rank() * dm_->segment_size() +
                 dm_->shape()[1] * p.second + p.first;
    if ((offset >= (int)dm_->data_size()) || (offset < 0)) {
      fmt::print("{}: index {} offset {} hb {} dm_off {} p=< {}, {} > \n",
                 default_comm().rank(), index_, offset,
                 dm_->halo_bounds().prev * dm_->shape()[1],
                 find_dm_offset(index_), p.first, p.second);
    }
    assert(offset > 0);
    assert(offset < (int)dm_->data_size());
    return *(dm_->data() + offset);
  }

  // friend operators fulfill rng::detail::weakly_equality_comparable_with_
  friend bool operator==(dm_subrange_iterator &first,
                         dm_subrange_iterator &second) {
    return first.index_ == second.index_;
  }
  friend bool operator!=(dm_subrange_iterator &first,
                         dm_subrange_iterator &second) {
    return first.index_ != second.index_;
  }
  friend bool operator==(dm_subrange_iterator first,
                         dm_subrange_iterator second) {
    return first.index_ == second.index_;
  }
  friend bool operator!=(dm_subrange_iterator first,
                         dm_subrange_iterator second) {
    return first.index_ != second.index_;
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

  bool is_local() { return dm_->is_local_cell(find_dm_offset(index_)); }

  // for debug purposes
  std::size_t find_dm_offset() const { return find_dm_offset(index_); }

// private:
  /*
   * converts index within subrange (viewed as linear contiguous space)
   * into index within physical segment in dm
   */
  std::size_t find_dm_offset(int index) const {
    int ind_rows, ind_cols;
    int offset = 0;

    ind_rows = index / (col_rng_.second - col_rng_.first);
    ind_cols = index % (col_rng_.second - col_rng_.first);

    // offset = dm_->halo_bounds().prev * dm_->shape()[1];
    offset += row_rng_.first * dm_->shape()[1] + col_rng_.first;
    offset += ind_rows * dm_->shape()[1] + ind_cols;

    return offset;
  };

// private:
  DM *dm_ = nullptr;
  std::pair<int, int> row_rng_ = std::pair<int, int>(0, 0);
  std::pair<int, int> col_rng_ = std::pair<int, int>(0, 0);

  std::size_t index_ = 0;
}; // class dm_subrange_iterator

template <typename DM> class dm_subrange {
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
  std::stringstream s;
  int _i = 0;
  s << default_comm().rank() << ": dm_transform ";
  for (mhp::dm_subrange_iterator<DM> i = in.begin(); i != in.end(); i++) {

    if (i.is_local()) {
      *(out) = op(i);
      s << _i << "(" << i.find_dm_offset() << ")" << *i << "->" << *(out) << "(" << out.index_ << "/" << out.find_dm_offset() << ")" << " \n";
    }
    ++out;
    _i++;
  }
  s << std::endl;
  std::cout << s.str();
}

} // namespace mhp
