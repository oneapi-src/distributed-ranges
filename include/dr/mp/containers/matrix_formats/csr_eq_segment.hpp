// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mp {

template <typename DSM> class csr_eq_segment_iterator;

template <typename DSM> class csr_eq_segment_reference {
  using iterator = csr_eq_segment_iterator<DSM>;

public:
  using value_type = typename DSM::value_type;
  using index_type = typename  DSM::index_type;
  using elem_type = typename  DSM::elem_type;

  csr_eq_segment_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  operator std::pair<std::pair<index_type, index_type>, elem_type>() const {
    return iterator_.get();
  }

  template <std::size_t Index> auto get() const noexcept {
    if constexpr (Index == 0) {
      return iterator_.get_index();
    }
    if constexpr (Index == 1) {
      return iterator_.get_value();
    }
  }

  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const csr_eq_segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // csr_eq_segment_reference

template <typename DSM> class csr_eq_segment_iterator {
public:
  using value_type = typename DSM::value_type;
  using index_type = typename  DSM::index_type;
  using elem_type = typename  DSM::elem_type;
  using difference_type = typename DSM::difference_type;

  csr_eq_segment_iterator() = default;
  csr_eq_segment_iterator(DSM *dsm, std::size_t segment_index, std::size_t index) {
    dsm_ = dsm;
    segment_index_ = segment_index;
    index_ = index;
  }

  auto operator<=>(const csr_eq_segment_iterator &other) const noexcept {
    // assertion below checks against compare dereferenceable iterator to a
    // singular iterator and against attempt to compare iterators from different
    // sequences like _Safe_iterator<gnu_cxx::normal_iterator> does
    assert(dsm_ == other.dsm_);
    return segment_index_ == other.segment_index_
               ? index_ <=> other.index_
               : segment_index_ <=> other.segment_index_;
  }

  // Comparison
  bool operator==(const csr_eq_segment_iterator &other) const noexcept {
    return (*this <=> other) == 0;
  }

  // Only this arithmetic manipulate internal state
  auto &operator+=(difference_type n) {
    assert(dsm_ != nullptr);
    assert(n >= 0 || static_cast<difference_type>(index_) >= -n);
    index_ += n;
    return *this;
  }

  auto &operator-=(difference_type n) { return *this += (-n); }

  difference_type operator-(const csr_eq_segment_iterator &other) const noexcept {
    assert(dsm_ != nullptr && dsm_ == other.dsm_);
    assert(index_ >= other.index_);
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
  friend auto operator+(difference_type n, const csr_eq_segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const {
    assert(dsm_ != nullptr);
    return csr_eq_segment_reference<DSM>{*this};
  }
  auto operator[](difference_type n) const {
    assert(dsm_ != nullptr);
    return *(*this + n);
  }

  void get(value_type *dst, std::size_t size) const {
    auto elems = new elem_type[size];
    auto indexes = new dr::index<index_type>[size];
    get_value(elems, size);
    get_index(indexes, size);
    for (std::size_t i = 0; i < size; i++) {
      *(dst + i) = {indexes[i], elems[i]};
    }
  }

  value_type get() const {
    value_type val;
    get(&val, 1);
    return val;
  }

  void get_value(elem_type *dst, std::size_t size) const {
    assert(dsm_ != nullptr);
    assert(segment_index_ * dsm_->segment_size_ + index_ < dsm_->nnz_);
    (dsm_->vals_data_->segments()[segment_index_].begin() + index_).get(dst, size);
  }
  
  elem_type get_value() const {
    elem_type val;
    get_value(&val, 1);
    return val;
  }

  void get_index(dr::index<index_type> *dst, std::size_t size) const {
    assert(dsm_ != nullptr);
    assert(segment_index_ * dsm_->segment_size_ + index_ < dsm_->nnz_);
    auto col_data = new index_type[size];
    (dsm_->cols_data_->segments()[segment_index_].begin() + index_).get(col_data, size);
    index_type *rows;
    std::size_t rows_length = dsm_->get_row_size(segment_index_);
    
    if (rank() == dsm_->rows_backend_.getrank()) {
      rows = dsm_->rows_data_;
    }
    else {
      rows = new index_type[rows_length];
      dsm_->rows_backend_.getmem(rows, 0, rows_length * sizeof(index_type), segment_index_);
    }
    auto position = dsm_->cols_data_->get_segment_offset(segment_index_) + index_;
    auto rows_iter = rows + 1;
    auto cols_iter = col_data;
    auto iter = dst;
    std::size_t current_row = dsm_->row_offsets_[segment_index_];
    std::size_t last_row = current_row + rows_length - 1;
    for (int i = 0; i < size; i++) {
      while (current_row < last_row && *rows_iter <= position + i ) {
        rows_iter++;
        current_row++;
      }
      iter->first = current_row;
      iter->second = *cols_iter;
      cols_iter++;
      iter++;
    }
    if (rank() != dsm_->rows_backend_.getrank()) {
      delete[] rows;
    }
    delete[] col_data;

  }

  dr::index<index_type> get_index() const {
    dr::index<index_type> val;
    get_index(&val, 1);
    return val;
  }

  void put(const value_type *dst, std::size_t size) const {
    assert(dsm_ != nullptr);
    assert(segment_index_ * dsm_->segment_size_ + index_ < dsm_->nnz_);
    (dsm_->vals_data_->segments()[segment_index_].begin() + index_).put(dst, size);
  }

  void put(const value_type &value) const { put(&value, 1); }

  auto rank() const {
    assert(dsm_ != nullptr);
    return segment_index_;
  }

  auto segments() const {
    assert(dsm_ != nullptr);
    return dr::__detail::drop_segments(dsm_->segments(), segment_index_, index_);
  }

private:
  // all fields need to be initialized by default ctor so every default
  // constructed iter is equal to any other default constructed iter
  DSM *dsm_ = nullptr;
  std::size_t segment_index_ = 0;
  std::size_t index_ = 0;
}; // csr_eq_segment_iterator

template <typename DSM> class csr_eq_segment {
private:
  using iterator = csr_eq_segment_iterator<DSM>;

public:
  using difference_type = std::ptrdiff_t;
  csr_eq_segment() = default;
  csr_eq_segment(DSM *dsm, std::size_t segment_index, std::size_t size,
             std::size_t reserved) {
    dsm_ = dsm;
    segment_index_ = segment_index;
    size_ = size;
    reserved_ = reserved;
    assert(dsm_ != nullptr);
  }

  auto size() const {
    assert(dsm_ != nullptr);
    return size_;
  }

  auto begin() const { return iterator(dsm_, segment_index_, 0); }
  auto end() const { return begin() + size(); }
  auto reserved() const { return reserved_; }

  auto operator[](difference_type n) const { return *(begin() + n); }

  bool is_local() const { return segment_index_ == default_comm().rank(); }

private:
  DSM *dsm_ = nullptr;
  std::size_t segment_index_;
  std::size_t size_;
  std::size_t reserved_;
}; // csr_eq_segment

} // namespace dr::mp

namespace std {
    template<typename DSM>
    struct tuple_size<dr::mp::csr_eq_segment_reference<DSM>> : std::integral_constant<std::size_t, 2> {};

    template <std::size_t Index, typename DSM>
    struct tuple_element<Index, dr::mp::csr_eq_segment_reference<DSM>>
        : tuple_element<Index, std::tuple<dr::index<typename DSM::index_type>, typename DSM::elem_type>> {};

} // namespace std
