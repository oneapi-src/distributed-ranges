// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


int some_id_base =0;
namespace dr::mp {
namespace __detail {
  template <typename T, typename V>
  class transform_fn_1 {
    public:
    using value_type = V;
    using index_type = T;
    transform_fn_1(std::size_t offset, std::size_t row_size, T* row_ptr):
      offset_(offset), row_size_(row_size), row_ptr_(row_ptr) {
      assert(offset_ == 0);
      myid = some_id_base++;
      fmt::print("created {}\n", myid);
      }

    ~transform_fn_1() {
      destroyed = true;
      fmt::print("destroyed {}\n", myid);
    }
    template <typename P>
    auto operator()(P entry) const {
      fmt::print("called {}\n", myid);
      assert(offset_ == 0);
      assert(!destroyed);
      auto [index, pair] = entry;
      auto [val, column] = pair;
      auto row = rng::distance(
                row_ptr_,
                std::upper_bound(row_ptr_, row_ptr_ + row_size_, offset_) -
                    1);
      dr::index<index_type> index_obj(row, column);
      value_type entry_obj(index_obj, val);
      return entry_obj;
    }
    private:
    int myid = 0;
    bool destroyed = false;
    std::size_t offset_;
    std::size_t row_size_;
    T* row_ptr_;
  };
}
template <typename DSM> class csr_row_segment_iterator;

template <typename DSM> class csr_row_segment_reference {
  using iterator = csr_row_segment_iterator<DSM>;

public:
  using value_type = typename DSM::value_type;
  using index_type = typename DSM::index_type;
  using elem_type = typename DSM::elem_type;

  csr_row_segment_reference(const iterator it) : iterator_(it) {}

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

  auto operator=(const csr_row_segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // csr_row_segment_reference

template <typename DSM> class csr_row_segment_iterator {
public:
  using value_type = typename DSM::value_type;
  using index_type = typename DSM::index_type;
  using elem_type = typename DSM::elem_type;
  using difference_type = typename DSM::difference_type;

  csr_row_segment_iterator() = default;
  csr_row_segment_iterator(DSM *dsm, std::size_t segment_index,
                           std::size_t index) {
    dsm_ = dsm;
    segment_index_ = segment_index;
    index_ = index;
    if (dsm_->vals_backend_.getrank() == segment_index_) {
      elem_view_ = get_elem_view(dsm_, segment_index);
      base_iter = elem_view_.begin();
    }
  }

  auto operator<=>(const csr_row_segment_iterator &other) const noexcept {
    // assertion below checks against compare dereferenceable iterator to a
    // singular iterator and against attempt to compare iterators from different
    // sequences like _Safe_iterator<gnu_cxx::normal_iterator> does
    assert(dsm_ == other.dsm_);
    return segment_index_ == other.segment_index_
               ? index_ <=> other.index_
               : segment_index_ <=> other.segment_index_;
  }

  // Comparison
  bool operator==(const csr_row_segment_iterator &other) const noexcept {
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

  difference_type
  operator-(const csr_row_segment_iterator &other) const noexcept {
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
  friend auto operator+(difference_type n,
                        const csr_row_segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const {
    assert(dsm_ != nullptr);
    return csr_row_segment_reference<DSM>{*this};
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
    dsm_->vals_backend_.getmem(dst, index_ * sizeof(elem_type),
                               size * sizeof(elem_type), segment_index_);
  }

  elem_type get_value() const {
    elem_type val;
    get_value(&val, 1);
    return val;
  }

  void get_index(dr::index<index_type> *dst, std::size_t size) const {
    assert(dsm_ != nullptr);
    assert(segment_index_ * dsm_->segment_size_ + index_ < dsm_->nnz_);
    index_type *col_data;
    if (rank() == dsm_->cols_backend_.getrank()) {
      col_data = dsm_->cols_data_ + index_;
    } else {
      col_data = new index_type[size];
      dsm_->cols_backend_.getmem(col_data, index_ * sizeof(index_type),
                                 size * sizeof(index_type), segment_index_);
    }
    index_type *rows;
    std::size_t rows_length = dsm_->segment_size_;
    rows = new index_type[rows_length];
    (dsm_->rows_data_->segments()[segment_index_].begin())
        .get(rows, rows_length);

    auto position = dsm_->val_offsets_[segment_index_] + index_;
    auto rows_iter = rows + 1;
    index_type *cols_iter = col_data;
    auto iter = dst;
    std::size_t current_row = dsm_->segment_size_ * segment_index_;
    std::size_t last_row =
        std::min(current_row + rows_length - 1, dsm_->shape_[0] - 1);

    for (int i = 0; i < size; i++) {
      while (current_row < last_row && *rows_iter <= position + i) {
        rows_iter++;
        current_row++;
      }
      iter->first = current_row;
      iter->second = *cols_iter;
      cols_iter++;
      iter++;
    }
    if (rank() != dsm_->cols_backend_.getrank()) {
      delete[] col_data;
    }
    delete[] rows;
  }

  dr::index<index_type> get_index() const {
    dr::index<index_type> val;
    get_index(&val, 1);
    return val;
  }

  auto rank() const {
    assert(dsm_ != nullptr);
    return segment_index_;
  }

  auto segments() const {
    assert(dsm_ != nullptr);
    return dr::__detail::drop_segments(dsm_->segments(), segment_index_,
                                       index_);
  }  

  auto local() const {
    const auto my_process_segment_index = dsm_->vals_backend_.getrank();
    assert(my_process_segment_index == segment_index_);
    auto [a, b] = *base_iter;
    auto [c, d] = a;
    fmt::print("aqwsedrftgyhuji {} {} {}\n", b, c, d);
    return base_iter;
  }

private:

  static auto get_elem_view(DSM *dsm, std::size_t segment_index) {
    std::size_t offset = dsm->segment_size_ * segment_index;
    auto row_size = dsm->segment_size_;
    auto vals_size = dsm->vals_size_;
    auto local_vals = dsm->vals_data_;
    auto local_vals_range = rng::subrange(local_vals, local_vals + vals_size);
    auto local_cols = dsm->cols_data_;
    auto local_cols_range = rng::subrange(local_cols, local_cols + vals_size);
    auto local_rows = dsm->rows_data_->segments()[segment_index].begin().local();
    auto zipped_results = rng::views::zip(local_vals_range, local_cols_range);
    auto enumerated_zipped = rng::views::enumerate(zipped_results);
    auto transformer = __detail::transform_fn_1<index_type, value_type>(offset, row_size, local_rows);
    return rng::views::transform(enumerated_zipped, transformer);
  }

  // all fields need to be initialized by default ctor so every default
  // constructed iter is equal to any other default constructed iter
  using view_type = decltype(get_elem_view(std::declval<DSM*>(), 0));
  using iter_type = rng::iterator_t<view_type>;
  view_type elem_view_;
  iter_type base_iter;
  DSM *dsm_ = nullptr;
  std::size_t segment_index_ = 0;
  std::size_t index_ = 0;
}; // csr_row_segment_iterator

template <typename DSM> class csr_row_segment {
private:
  using iterator = csr_row_segment_iterator<DSM>;

public:
  using difference_type = std::ptrdiff_t;
  csr_row_segment() = default;
  csr_row_segment(DSM *dsm, std::size_t segment_index, std::size_t size,
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
}; // csr_row_segment

} // namespace dr::mp

namespace std {
template <typename DSM>
struct tuple_size<dr::mp::csr_row_segment_reference<DSM>>
    : std::integral_constant<std::size_t, 2> {};

template <std::size_t Index, typename DSM>
struct tuple_element<Index, dr::mp::csr_row_segment_reference<DSM>>
    : tuple_element<Index, std::tuple<dr::index<typename DSM::index_type>,
                                      typename DSM::elem_type>> {};

} // namespace std
