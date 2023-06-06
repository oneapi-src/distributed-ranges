// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <dr/mhp/containers/distribution.hpp>

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
    assert(dv_ != nullptr);
    const auto my_process_segment_index = dv_->win_.communicator().rank();

    if (my_process_segment_index == segment_index_)
      return dv_->data_ + index_ + dv_->distribution_.halo().prev;

    assert(!dv_->distribution_.halo().periodic); // not implemented

    // sliding view needs local iterators that point to the halo
    if (my_process_segment_index + 1 == segment_index_) {
      assert(index_ <= dv_->distribution_.halo()
                           .next); // <= instead of < to cover end() case
      return dv_->data_ + dv_->distribution_.halo().prev + index_ +
             dv_->segment_size_;
    }

    if (my_process_segment_index == segment_index_ + 1) {
      assert(dv_->segment_size_ - index_ <= dv_->distribution_.halo().prev);
      return dv_->data_ + dv_->distribution_.halo().prev + index_ -
             dv_->segment_size_;
    }

    assert(false); // trying to read non-owned memory
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

/// distributed vector
template <typename T, typename Allocator = dr::mhp::default_allocator<T>>
class distributed_vector {

public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using allocator_type = Allocator;

  class iterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename distributed_vector::value_type;
    using difference_type = typename distributed_vector::difference_type;

    iterator() {}
    iterator(const distributed_vector *parent, difference_type offset)
        : parent_(parent), offset_(offset) {}

    auto operator+(difference_type n) const {
      return iterator(parent_, offset_ + n);
    }
    friend auto operator+(difference_type n, const iterator &other) {
      return other + n;
    }
    auto operator-(difference_type n) const {
      return iterator(parent_, offset_ - n);
    }
    auto operator-(iterator other) const { return offset_ - other.offset_; }

    auto &operator+=(difference_type n) {
      offset_ += n;
      return *this;
    }
    auto &operator-=(difference_type n) {
      offset_ -= n;
      return *this;
    }
    auto &operator++() {
      offset_++;
      return *this;
    }
    auto operator++(int) {
      auto old = *this;
      offset_++;
      return old;
    }
    auto &operator--() {
      offset_--;
      return *this;
    }
    auto operator--(int) {
      auto old = *this;
      offset_--;
      return old;
    }

    bool operator==(iterator other) const {
      if (parent_ == nullptr || other.parent_ == nullptr) {
        return false;
      } else {
        return offset_ == other.offset_;
      }
    }
    auto operator<=>(iterator other) const {
      assert(parent_ == other.parent_);
      return offset_ <=> other.offset_;
    }

    auto operator*() const {
      auto segment_size = parent_->segment_size_;
      return parent_
          ->segments()[offset_ / segment_size][offset_ % segment_size];
    }
    auto operator[](difference_type n) const { return *(*this + n); }

    auto local() {
      auto segment_size = parent_->segment_size_;
      return (parent_->segments()[offset_ / segment_size].begin() +
              offset_ % segment_size)
          .local();
    }

    //
    // Support for distributed ranges
    //
    // distributed iterator provides segments
    // remote iterator provides local
    //
    auto segments() {
      return dr::__detail::drop_segments(parent_->segments(), offset_);
    }

  private:
    const distributed_vector *parent_ = nullptr;
    difference_type offset_;
  };

  // Do not copy
  // We need a move constructor for the implementation of reduce algorithm
  distributed_vector(const distributed_vector &) = delete;
  distributed_vector &operator=(const distributed_vector &) = delete;
  distributed_vector(distributed_vector &&) { assert(false); }

  /// Constructor
  distributed_vector(std::size_t size = 0, distribution dist = distribution()) {
    init(size, dist, Allocator());
  }

  /// Constructor
  distributed_vector(std::size_t size, value_type fill_value,
                     distribution dist = distribution()) {
    init(size, dist, Allocator());
    mhp::fill(*this, fill_value);
  }

  ~distributed_vector() {
    fence();
    active_wins().erase(win_.mpi_win());
    win_.free();
    allocator_.deallocate(data_, data_size_);
    data_ = nullptr;
    delete halo_;
  }

  /// Returns iterator to beginning
  auto begin() const { return iterator(this, 0); }
  /// Returns iterator to end
  auto end() const { return begin() + size_; }

  /// Returns size
  auto size() const { return size_; }
  /// Returns reference using index
  auto operator[](difference_type n) const { return *(begin() + n); }
  auto &halo() { return *halo_; }

  auto segments() const { return rng::views::all(segments_); }

  void dump_to_log(std::string desc = "") const {
    dr::drlog.debug("DV {}, dataAddr:{}", desc,
                    static_cast<void *>(this->data_));
    for (std::size_t idx = 0; idx < data_size_; ++idx)
      dr::drlog.debug(" - idx:{} val:{} addr:{}", idx, this->data_[idx],
                      static_cast<void *>(this->data_ + idx));
    dr::drlog.debug("\n");
  }

private:
  void init(auto size, auto dist, const auto &allocator) {
    allocator_ = allocator;
    size_ = size;
    distribution_ = dist;

    // determine the distribution of data
    auto comm_size = default_comm().size(); // dr-style ignore
    auto hb = dist.halo();
    std::size_t gran = dist.granularity();
    // TODO: make this an error that is reported back to user
    assert(size % gran == 0 && "size must be a multiple of the granularity");
    assert(hb.prev % gran == 0 && "size must be a multiple of the granularity");
    assert(hb.next % gran == 0 && "size must be a multiple of the granularity");
    segment_size_ = gran * std::max({(size / gran + comm_size - 1) / comm_size,
                                     hb.prev / gran, hb.next / gran});

    data_size_ = segment_size_ + hb.prev + hb.next;
    if (size_ > 0) {
      data_ = allocator_.allocate(data_size_);
    }

    halo_ = new span_halo<T>(default_comm(), data_, data_size_, hb);

    std::size_t segment_index = 0;
    for (std::size_t i = 0; i < size; i += segment_size_) {
      segments_.emplace_back(this, segment_index++,
                             std::min(segment_size_, size - i));
    }

    win_.create(default_comm(), data_, data_size_ * sizeof(T));
    active_wins().insert(win_.mpi_win());
    fence();
  }

  friend dv_segment_iterator<distributed_vector>;

  std::size_t segment_size_ = 0;
  std::size_t data_size_ = 0;
  T *data_ = nullptr;
  span_halo<T> *halo_;

  distribution distribution_;
  std::size_t size_;
  std::vector<dv_segment<distributed_vector>> segments_;
  dr::rma_window win_;
  Allocator allocator_;
};

template <typename DR>
concept has_halo_method = dr::distributed_range<DR> && requires(DR &&dr) {
  { rng::begin(dr::ranges::segments(dr)[0]).halo() };
};

auto &halo(has_halo_method auto &&dr) {
  return rng::begin(dr::ranges::segments(dr)[0]).halo();
}

} // namespace dr::mhp
