// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace dr::mhp {

template <typename DV> class dv_segment_iterator;

template <typename DV> class dv_segment_reference {
  using iterator = dv_segment_iterator<DV>;

public:
  using value_type = typename DV::value_type;

  dv_segment_reference(const iterator it) : iterator_(it) {}

  operator value_type() const { return iterator_.get(); }
  auto operator=(const value_type &value) const {
    iterator_.put(value);
    return *this;
  }
  auto operator=(const dv_segment_reference &other) const {
    *this = value_type(other);
    return *this;
  }
  auto operator&() const { return iterator_; }

private:
  const iterator iterator_;
}; // dv_segment_reference

template <typename DV> class dv_segment_iterator {
public:
  using value_type = typename DV::value_type;
  using size_type = typename DV::size_type;
  using difference_type = typename DV::difference_type;

  dv_segment_iterator() = default;
  dv_segment_iterator(DV *dv, std::size_t segment_index, std::size_t index) {
    dv_ = dv;
    segment_index_ = segment_index;
    index_ = index;
  }

  auto operator<=>(const dv_segment_iterator &other) const noexcept {
    // assertion below checks against compare dereferenceable iterator to a
    // singular iterator and against attempt to compare iterators from different
    // sequences like _Safe_iterator<gnu_cxx::normal_iterator> does
    assert(dv_ == other.dv_);
    return segment_index_ == other.segment_index_
               ? index_ <=> other.index_
               : segment_index_ <=> other.segment_index_;
  }

  // Comparison
  bool operator==(const dv_segment_iterator &other) const noexcept {
    return (*this <=> other) == 0;
  }

  // Only this arithmetic manipulate internal state
  auto &operator+=(difference_type n) {
    assert(dv_ != nullptr);
    assert(n >= 0 || static_cast<difference_type>(index_) >= -n);
    index_ += n;
    return *this;
  }

  auto &operator-=(difference_type n) { return *this += (-n); }

  difference_type operator-(const dv_segment_iterator &other) const noexcept {
    assert(dv_ != nullptr && dv_ == other.dv_);
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
  friend auto operator+(difference_type n, const dv_segment_iterator &other) {
    return other + n;
  }

  // dereference
  auto operator*() const {
    assert(dv_ != nullptr);
    return dv_segment_reference<DV>{*this};
  }
  auto operator[](difference_type n) const {
    assert(dv_ != nullptr);
    return *(*this + n);
  }

  void get(value_type *dst, std::size_t size) const {
    assert(dv_ != nullptr);
    assert(segment_index_ * dv_->segment_size_ + index_ < dv_->size_);
    auto segment_offset = index_ + dv_->distribution_.halo().prev;
    dv_->win_.get(dst, size * sizeof(*dst), segment_index_,
                  segment_offset * sizeof(*dst));
  }

  value_type get() const {
    value_type val;
    get(&val, 1);
    return val;
  }

  void put(const value_type *dst, std::size_t size) const {
    assert(dv_ != nullptr);
    assert(segment_index_ * dv_->segment_size_ + index_ < dv_->size_);
    auto segment_offset = index_ + dv_->distribution_.halo().prev;
    dr::drlog.debug("dv put:: ({}:{}:{})\n", segment_index_, segment_offset,
                    size);
    dv_->win_.put(dst, size * sizeof(*dst), segment_index_,
                  segment_offset * sizeof(*dst));
  }

  void put(const value_type &value) const { put(&value, 1); }

  auto rank() const {
    assert(dv_ != nullptr);
    return segment_index_;
  }

  auto local() const {
#ifndef SYCL_LANGUAGE_VERSION
    assert(dv_ != nullptr);
#endif
    const auto my_process_segment_index = dv_->win_.communicator().rank();

    if (my_process_segment_index == segment_index_)
      return dv_->data_ + index_ + dv_->distribution_.halo().prev;
#ifndef SYCL_LANGUAGE_VERSION
    assert(!dv_->distribution_.halo().periodic); // not implemented
#endif
    // sliding view needs local iterators that point to the halo
    if (my_process_segment_index + 1 == segment_index_) {
#ifndef SYCL_LANGUAGE_VERSION
      assert(index_ <= dv_->distribution_.halo()
                           .next); // <= instead of < to cover end() case
#endif
      return dv_->data_ + dv_->distribution_.halo().prev + index_ +
             dv_->segment_size_;
    }

    if (my_process_segment_index == segment_index_ + 1) {
#ifndef SYCL_LANGUAGE_VERSION
      assert(dv_->segment_size_ - index_ <= dv_->distribution_.halo().prev);
#endif
      return dv_->data_ + dv_->distribution_.halo().prev + index_ -
             dv_->segment_size_;
    }

#ifndef SYCL_LANGUAGE_VERSION
    assert(false); // trying to read non-owned memory
#endif
    return static_cast<decltype(dv_->data_)>(nullptr);
  }

  auto segments() const {
    assert(dv_ != nullptr);
    return dr::__detail::drop_segments(dv_->segments(), segment_index_, index_);
  }

  auto &halo() const {
    assert(dv_ != nullptr);
    return dv_->halo();
  }
  auto halo_bounds() const {
    assert(dv_ != nullptr);
    return dv_->distribution_.halo();
  }

private:
  // all fields need to be initialized by default ctor so every default
  // constructed iter is equal to any other default constructed iter
  DV *dv_ = nullptr;
  std::size_t segment_index_ = 0;
  std::size_t index_ = 0;
}; // dv_segment_iterator

template <typename DV> class dv_segment {
private:
  using iterator = dv_segment_iterator<DV>;

public:
  using difference_type = std::ptrdiff_t;
  dv_segment() = default;
  dv_segment(DV *dv, std::size_t segment_index, std::size_t size) {
    dv_ = dv;
    segment_index_ = segment_index;
    size_ = size;
    assert(dv_ != nullptr);
  }

  auto size() const {
    assert(dv_ != nullptr);
    return size_;
  }

  auto begin() const { return iterator(dv_, segment_index_, 0); }
  auto end() const { return begin() + size(); }

  auto operator[](difference_type n) const { return *(begin() + n); }

private:
  DV *dv_ = nullptr;
  std::size_t segment_index_;
  std::size_t size_;
}; // dv_segment
